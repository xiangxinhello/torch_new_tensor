#include <ATen/NamedTensorUtils.h>
#include <c10/core/DeviceType.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/core/impl/HermeticPyObjectTLS.h>
#include <c10/core/impl/PythonDispatcherTLS.h>
#include <c10/util/irange.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/Size.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/Types.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/python_cpp_function.h>
#include <torch/csrc/autograd/python_hook.h>
#include <torch/csrc/autograd/python_variable_indexing.h>
#include <torch/csrc/autograd/utils/error_messages.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/pyobject_preservation.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/python_dispatch.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_numpy.h>

#include <torch/csrc/utils/torch_dispatch_mode.h>

#include <ATen/ATen.h>

#include <c10/core/SymIntArrayRef.h>
#include <structmember.h>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

using namespace at;
using namespace torch;
using namespace torch::autograd;


PyTypeObject THPVariableType = {
    PyVarObject_HEAD_INIT(
        &THPVariableMetaType,
        0) "torch._C.TensorBase", /* tp_name */
    sizeof(THPVariable), /* tp_basicsize */
    0, /* tp_itemsize */
    // This is unspecified, because it is illegal to create a THPVariableType
    // directly.  Subclasses will have their tp_dealloc set appropriately
    // by the metaclass
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    &THPVariable_as_mapping, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    // NOLINTNEXTLINE(misc-redundant-expression)
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE |
        Py_TPFLAGS_HAVE_GC, /* tp_flags */
    nullptr, /* tp_doc */
    // Also set by metaclass
    (traverseproc)THPFunction_traverse, /* tp_traverse */
    (inquiry)THPVariable_clear, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    nullptr, /* tp_methods */
    nullptr, /* tp_members */
    THPVariable_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    // Although new is provided here, it is illegal to call this with cls ==
    // THPVariableMeta.  Instead, subclass it first and then construct it
    THPVariable_pynew, /* tp_new   */
};

PyObject* THPVariable_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      type != &THPVariableType,
      "Cannot directly construct TensorBase; subclass it and then construct that");
  jit::tracer::warn("torch.Tensor", jit::tracer::WARN_CONSTRUCTOR);
  auto tensor = torch::utils::base_tensor_ctor(args, kwargs);
  // WARNING: tensor is NOT guaranteed to be a fresh tensor; e.g., if it was
  // given a raw pointer that will refcount bump
  // NB: base_tensor_ctor can call into dispatched ATen functions (e.g.,
  // alias(), lift_fresh()) which can return Tensor subclasses.  We allow
  // these to be passed on directly.
  return THPVariable_NewWithVar(
      type,
      std::move(tensor),
      c10::impl::PyInterpreterStatus::MAYBE_UNINITIALIZED,
      /*allow_preexisting_pyobj=*/true);
  END_HANDLE_TH_ERRORS
}


// Creates a new Python object for a Variable.  The status parameter
// specifies what the interpreter tag status on the object is; for
// example, if you ran check_pyobj, the return optional of this object
// tells you if the tensor was already tagged or not so you can pass
// TAGGED_BY_US or MAYBE_UNINITIALIZED; in other cases, you know where
// var came from and can directly assert that it's DEFINITELY_UNINITIALIZED.
// It's ALWAYS safe (albeit slower) to call this with MAYBE_UNINITIALIZED.
static PyObject* THPVariable_NewWithVar(
    PyTypeObject* type,
    Variable _var,
    c10::impl::PyInterpreterStatus status,
    bool allow_preexisting_pyobj) {
  // Make sure that the reinterpret into a THPVariable* will be valid
  TORCH_CHECK(
      PyType_IsSubtype(type, &THPVariableType),
      "Creating a Tensor subclass from a class ",
      "that does not inherit from Tensor is not possible. Make sure your class inherits from Tensor.");

  // This function overwrite the Tensor's pyobj field without extra checks
  // Make sure it is not set otherwise we would leak memory
  auto mb_obj = _var.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(
      getPyInterpreter(), /*ignore_hermetic_tls=*/false);

  // Under some circumstances, we may attempt to create a new Python
  // object for a variable that already has a Python object.  The most common
  // situation this can occur is if you have a TorchDispatchMode active that
  // is returning a subclass from lift_fresh (which is invoked to
  // appropriately "wrap" a constant tensor into whatever ambient modes are
  // active.)
  //
  // In general, it is impossible to handle this case compositionally.
  // Suppose you have a user call ATensor([1, 2, 3]) when a mode is active
  // that is transforming all ops (including the internal lift_fresh call that
  // transforms [1, 2, 3] into a torch.tensor([1., 2., 3.])) to output
  // BTensor, where ATensor and BTensor are completely unrelated subclasses
  // and there is no way to compose them.  There is no way to satisfy the user
  // request here: in particular, you can't just try to re-invoke the ATensor
  // constructor on the returned BTensor, because (1) this could cause an
  // infinite loop--we are already in ATensor.__new__ and (2) there isn't any
  // guarantee that ATensor.__new__ supports a single element constructor
  // anyway.
  //
  // However, a more common case is a user just called torch.Tensor([1, 2, 3]),
  // and a fake tensor mode is active.  Really, all you want is to get back
  // a FakeTensor, in the same way torch.tensor([1, 2, 3]) or torch.arange(3)
  // would have returned a fake tensor (concretely, the way this happens
  // is we create a *real* tensor torch.tensor([1., 2., 3.]), and then it
  // turns into a FakeTensor when we call lift_fresh on this real tensor).
  // This case is compositional because FakeTensor is a subclass of Tensor, so
  // it's valid for us to return it in place of a Tensor.  So this is what we
  // do.

  if (mb_obj.has_value() && mb_obj.value()) {
    TORCH_CHECK(
        allow_preexisting_pyobj,
        "Creating a new Tensor subclass ",
        type->tp_name,
        " but the raw Tensor object is already associated to a python object ",
        "of type ",
        mb_obj.value()->ob_type->tp_name);
    // Even if we allow pre-existing PyObject, we don't allow completely
    // ignoring the requested type.  Check that we fulfilled a subtype
    // relation here.  In the common case the requested type is Tensor and
    // this always succeeds.
    PyObject* obj = *mb_obj;
    // Check if it's OK to just directly return the Python object without
    // allocating a new variable.  We just check that the existing Python
    // object is a subclass of the requested type.
    PyTypeObject* obj_type = Py_TYPE(obj);
    TORCH_CHECK(
        obj_type == type || PyType_IsSubtype(obj_type, type),
        "Creating a new Tensor subclass ",
        type->tp_name,
        " but the raw Tensor object is already associated to a python object ",
        "of type ",
        mb_obj.value()->ob_type->tp_name,
        " which is not a subclass of the "
        "requested type");
    // We may (in fact, we typically will) need to resurrect this
    return THPVariable_Wrap(std::move(_var));
  }

  PyObject* obj = type->tp_alloc(type, 0);
  if (obj) {
    auto v = (THPVariable*)obj;
    // TODO: named constructor to avoid default initialization
    new (&v->cdata) MaybeOwned<Variable>();
    if (c10::impl::HermeticPyObjectTLS::get_state()) {
      // Do NOT initialize pyobj field on the tensor, you own the C++
      v->cdata = MaybeOwned<Variable>::owned(std::move(_var));
      TORCH_INTERNAL_ASSERT(
          !check_has_torch_dispatch(obj),
          "While HermeticPyObject was enabled, we attempted to create a tensor "
          "subclass with __torch_dispatch__.  This violates the invariant that "
          "operations in HermeticPyObject have equivalent C++ implementations. "
          "If your operator registered from Python operator registration isn't "
          "doing anything strange, there may be an internal PyTorch bug involving "
          "not appropriately disabling TorchDispatchMode before executing "
          "Python op registration.");
    } else {
      // Normal codepath
      v->cdata = MaybeOwned<Variable>::owned(std::move(_var));
      const auto& var = THPVariable_Unpack(v);
      var.unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(
          getPyInterpreter(), obj, status);
      if (check_has_torch_dispatch(obj)) {
        var.unsafeGetTensorImpl()->set_python_dispatch(true);
      }
    }
  }
  return obj;
}



bool THPVariable_initModule(PyObject* module) {
  THPVariableMetaType.tp_base = &PyType_Type;
  if (PyType_Ready(&THPVariableMetaType) < 0)
    return false;
  Py_INCREF(&THPVariableMetaType);
  PyModule_AddObject(module, "_TensorMeta", (PyObject*)&THPVariableMetaType);

  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "TensorBase", (PyObject*)&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  torch::autograd::initTorchFunctions(module);
  torch::autograd::initTensorImplConversion(module);
  torch::utils::validate_numpy_for_dlpack_deleter_bug();
  return true;
}
