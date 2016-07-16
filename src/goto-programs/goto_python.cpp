#ifdef WITH_PYTHON

#include "goto_functions.h"

#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

class dummy_goto_class { };

static boost::python::list
get_instructions(const goto_programt &prog)
{
  using namespace boost::python;

  list pylist;
  std::vector<object> py_obj_vec;
  std::set<goto_programt::const_targett> targets;
  std::map<goto_programt::const_targett, unsigned int> target_map;

  // Convert instructions into python objects -- store in python list, as well
  // as in an stl vector, for easy access by index. Collect a set of all the
  // target iterators that are used in this function as well.
  for (const goto_programt::instructiont &insn : prog.instructions) {
    object o(insn);
    pylist.append(o);
    py_obj_vec.push_back(o);

    if (!insn.targets.empty()) {
      assert(insn.targets.size() == 1 && "Insn with multiple targets");
      targets.insert(*insn.targets.begin());
    }
  }

  // Map target iterators to index positions in the instruction list. Their
  // positions is the structure that we'll map over to python.
  unsigned int i = 0;
  for (auto it = prog.instructions.begin();
       it != prog.instructions.end();
       it++, i++) {
    if (targets.find(it) != targets.end())
      target_map.insert(std::make_pair(it, i));
  }

  // Iterate back over all the instructions again, this time filling out the
  // target attribute for each corresponding python object. If there's no
  // target, set it to None, otherwise set it to a reference to the
  // corresponding other python object.
  i = 0;
  for (const goto_programt::instructiont &insn : prog.instructions) {
    if (insn.targets.empty()) {
      // If there's no target, set the target attribute to None
      py_obj_vec[i].attr("target") = object();
    } else {
      assert(insn.targets.size() == 1 && "Insn with multiple targets");
      auto it = *insn.targets.begin();
      auto target_it = target_map.find(it);
      assert(target_it != target_map.end());

      // Set target attr to be reference to the correspondingly indexed python
      // object.
      py_obj_vec[i].attr("target") = py_obj_vec[target_it->second];
    }
    i++;
  }

  return pylist;
}

static void
set_instructions(const goto_programt &prog)
{
  (void)prog;
}


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
    .def_readwrite("target_number", &insnt::target_number);
  // No access here to the 'targets' field, see below.

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
    .def("set_instructions", &set_instructions);
}

#endif
