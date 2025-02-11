#include <OpenReg.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <ATen/Parallel.h>
#include <vector>

// Make this a proper CPython module
static struct PyModuleDef openreg_C_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pytorch_openreg._C",
};

PyObject* initModule() {
  PyObject* mod = PyModule_Create(&openreg_C_module);
  
  at::internal::lazy_init_num_threads();
  THPVariable_initModule(module);
  
  return mod;
}

