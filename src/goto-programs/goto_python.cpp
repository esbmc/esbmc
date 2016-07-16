#ifdef WITH_PYTHON

#include "goto_functions.h"

#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

class dummy_goto_class { };

void
build_goto_func_class()
{
  using namespace boost::python;

  scope types = class_<dummy_goto_class>("goto_programs");

  // Register relevant portions of goto functions / programs structure.
  class_<goto_functionst::function_mapt>("function_mapt", no_init)
    .def(map_indexing_suite<goto_functionst::function_mapt>());
  class_<goto_functionst>("goto_functionst", no_init)
    .def_readwrite("function_map", &goto_functionst::function_map);

  class_<goto_functiont>("goto_functiont")
    .def_readwrite("body", &goto_functiont::body)
    // No old-typet definitions available
    .def_readwrite("type", &goto_functiont::type)
    .def_readwrite("body_available", &goto_functiont::body_available)
    // No easy def for inlined_funcs
    .def_readwrite("inlined_funcs", &goto_functiont::inlined_funcs);
}

#endif
