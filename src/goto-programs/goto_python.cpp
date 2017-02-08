#ifdef WITH_PYTHON

#include <sstream>

#include "goto_functions.h"

#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

class dummy_goto_class { };

static bool
insn_lt(const goto_programt::instructiont &i1, const goto_programt::instructiont &i2)
{
  return i1.location_number < i2.location_number;
}

boost::python::object
get_instructions(const goto_programt &prog)
{
  using namespace boost::python;

  list l;

  auto listappend = [](list &li, object &o) {
    li.append(o);
  };

  auto setattr = [](object &o, object &target) {
    o.attr("target") = target;
  };

  auto setnone = [](object &o) {
    o.attr("target") = object();
  };

  prog.extract_instructions
    <list, decltype(listappend), object, decltype(setattr), decltype(setnone)>
    (l, listappend, setattr, setnone);

  return l;
}

void
set_instructions(goto_programt &prog, boost::python::object o)
{
  using namespace boost::python;

  list pylist = extract<list>(o);

  auto fetchelem = [](list &li, unsigned int idx) {
    return li[idx];
  };

  auto elemtoinsn = [](object &o) {
    goto_programt::instructiont insn = extract<goto_programt::instructiont>(o);
    return insn;
  };

  auto getattr = [](object &o) {
    return o.attr("target");
  };

  auto isattrnil = [](object &&o) {
    return o.is_none();
  };

  prog.inject_instructions
    <list, object, decltype(fetchelem), decltype(elemtoinsn), decltype(getattr), decltype(isattrnil)>
    (pylist, len(pylist), fetchelem, elemtoinsn, getattr, isattrnil);
  return;
}

extern namespacet *pythonctx_ns;

std::string
prog_to_string(const goto_programt &prog)
{
  std::stringstream ss;
  assert(pythonctx_ns != NULL);
  prog.output(*pythonctx_ns, "*", ss);
  return ss.str();
}

std::string
insn_to_string(const goto_programt::instructiont &insn,
    bool show_location = true, bool show_variables = false)
{
  // Stuff this insn in a list and feed it to output_instruction.
  goto_programt::instructionst list;
  std::stringstream ss;
  contextt ctx;

  assert(pythonctx_ns != NULL);
  insn.output_instruction(*pythonctx_ns, "", ss, show_location, show_variables);

  return ss.str();
}

void
build_goto_func_class()
{
  using namespace boost::python;

  scope esbmc;

  object progs(handle<>(borrowed(PyImport_AddModule("esbmc.goto_programs"))));
  scope quux = progs;

  esbmc.attr("goto_programs") = progs;

  // Register relevant portions of goto functions / programs structure.
  class_<goto_functionst::function_mapt>("function_mapt")
    .def(map_indexing_suite<goto_functionst::function_mapt>());
  class_<goto_functionst>("goto_functionst")
    .def_readwrite("function_map", &goto_functionst::function_map)
    .def("update", &goto_functionst::update);

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

  typedef goto_programt::instructiont insnt;
  class_<insnt>("instructiont")
    .def_readwrite("code", &insnt::code)
    .def_readwrite("function", &insnt::function)
    .def_readwrite("location", &insnt::location)
    .def_readwrite("type", &insnt::type)
    .def_readwrite("guard", &insnt::guard)
    // No list wrapper right now
    .def_readwrite("labels", &insnt::labels)
    // Skip k-inductoin stuff
    .def_readwrite("location_number", &insnt::location_number)
    .def_readwrite("loop_number", &insnt::loop_number)
    .def_readwrite("target_number", &insnt::target_number)
  // No access here to the 'targets' field, see below.
    .def("to_string", &insn_to_string, (arg("this"), arg("show_location")=false,
                                        arg("show_variables")=false))
    .def("clear", &insnt::clear, (arg("this"),
                            arg("type")=goto_program_instruction_typet::SKIP))
    .def("is_goto", &insnt::is_goto)
    .def("is_return", &insnt::is_return)
    .def("is_assign", &insnt::is_assign)
    .def("is_function_call", &insnt::is_function_call)
    .def("is_throw", &insnt::is_throw)
    .def("is_catch", &insnt::is_catch)
    .def("is_skip", &insnt::is_skip)
    .def("is_location", &insnt::is_location)
    .def("is_other", &insnt::is_other)
    .def("is_assume", &insnt::is_other)
    .def("is_assert", &insnt::is_assert)
    .def("is_atomic_begin", &insnt::is_atomic_begin)
    .def("is_atomic_end", &insnt::is_atomic_end)
    .def("is_end_function", &insnt::is_end_function)
    // Define a less-than operator so that we can put instructions in maps,
    // based on their location number.
    .def("__lt__", &insn_lt);
  // Can't publish "is backwards goto" because it touches targets

  // Trickyness: the 'targets' field of an instruction is very well suited,
  // containing an iterator to the instructiont that the current instruction
  // branches to. This doesn't translate to python in any way though. So: I
  // reckon we can construct a parallel representation in python, which doesn't
  // involve converting iterators to or from anything. We convert the
  // instructiont without the 'targets' field in to a python list, and then
  // for each non-empty targets entry, add to that the object instance
  // dictionary as a reference to the _python_ instructiont instance. The same
  // relationship is built, but with python objs.
  // Build this situation with an explicit getter and setter methods for the
  // goto_programt class: this is to remind the operator that they're
  // duplicating the instruciton list out of esbmc, and have to set it back in
  // for it to have an effect.
  class_<goto_programt>("goto_programt")
    // NB: these are not member methods, but pythons name resolution treats
    // them as if they were.
    .def("get_instructions", &get_instructions)
    .def("set_instructions", &set_instructions)
    // Most of the instruction manipulation methods of this class are subsumed
    // by the ability to treat it as a list of instructions in python.
    .def("to_string", &prog_to_string)
    .def("update", &goto_programt::update)
    .def("clear", &goto_programt::clear)
    .def("empty", &goto_programt::empty);
}

#endif
