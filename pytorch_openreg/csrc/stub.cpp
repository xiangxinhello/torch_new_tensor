#include <pybind11/pybind11.h>

extern PyObject* initModule();

PyMODINIT_FUNC PyInit__C(void) {
  return initModule();
}
