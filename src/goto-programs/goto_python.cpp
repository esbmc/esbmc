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

  enum_<goto_program_instruction_typet>("goto_program_instruction_type")
    .value("NO_INSTRUCTION_TYPE", goto_program_instruction_typet::NO_INSTRUCTION_TYPE)
    .value("GOTO", goto_program_instruction_typet::GOTO)
    .value("ASSUME", goto_program_instruction_typet::ASSUME)
    .value("ASSERT", goto_program_instruction_typet::ASSERT)
    .value("OTHER", goto_program_instruction_typet::OTHER)
    .value("SKIP", goto_program_instruction_typet::SKIP)
    .value("LOCATION", goto_program_instruction_typet::LOCATION)
    .value("END_FUNCTION", goto_program_instruction_typet::END_FUNCTION)
    .value("ATOMIC_BEGIN", goto_program_instruction_typet::ATOMIC_BEGIN)
    .value("ATOMIC_END", goto_program_instruction_typet::ATOMIC_END)
    .value("RETURN", goto_program_instruction_typet::RETURN)
    .value("ASSIGN", goto_program_instruction_typet::ASSIGN)
    .value("DECL", goto_program_instruction_typet::DECL)
    .value("DEAD", goto_program_instruction_typet::DEAD)
    .value("FUNCTION_CALL", goto_program_instruction_typet::FUNCTION_CALL)
    .value("THROW", goto_program_instruction_typet::THROW)
    .value("CATCH", goto_program_instruction_typet::CATCH)
    .value("THROW_DECL", goto_program_instruction_typet::THROW_DECL)
    .value("THROW_DECL_END", goto_program_instruction_typet::THROW_DECL_END);
}

#endif
