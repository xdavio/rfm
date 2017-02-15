#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "fm_py.h"

namespace py = pybind11;

PYBIND11_PLUGIN(c_fm) {
    py::module m("c_fm", R"pbdoc(
        c_fm
        ------
        docstring for c_fm module

    )pbdoc");

    m.def("fit_fm", &c_fit_fm, R"pbdoc(

    docstring goes here

    )pbdoc");

    m.def("predictfm", &c_predictfm, R"pbdoc(

    docstring goes here

    )pbdoc");
    
#ifdef VERSION_INFO
    m.attr("__version__") = py::str(VERSION_INFO);
#else
    m.attr("__version__") = py::str("dev");
#endif

    return m.ptr();
}
