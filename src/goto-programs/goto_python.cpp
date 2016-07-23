#ifdef WITH_PYTHON

#include <sstream>

#include "goto_functions.h"

#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

class dummy_goto_class { };

template <typename OutList, typename ListAppender, typename OutElem,
         typename SetAttrObj, typename SetAttrNil>
void
get_instructions_templ(const goto_programt &prog, OutList &list,
    ListAppender listappend, SetAttrObj setattrobj, SetAttrNil setattrnil)
{
  std::vector<OutElem> py_obj_vec;
  std::set<goto_programt::const_targett> targets;
  std::map<goto_programt::const_targett, unsigned int> target_map;

  // Convert instructions into python objects -- store in python list, as well
  // as in an stl vector, for easy access by index. Collect a set of all the
  // target iterators that are used in this function as well.
  for (const goto_programt::instructiont &insn : prog.instructions) {
    OutElem o(insn);
    listappend(list, o);
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
      setattrnil(py_obj_vec[i]);
    } else {
      assert(insn.targets.size() == 1 && "Insn with multiple targets");
      auto it = *insn.targets.begin();
      auto target_it = target_map.find(it);
      assert(target_it != target_map.end());

      // Set target attr to be reference to the correspondingly indexed python
      // object.
      setattrobj(py_obj_vec[i], py_obj_vec[target_it->second]);
    }
    i++;
  }

  return;
}

void
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

  get_instructions_templ<list, decltype(listappend), object, decltype(setattr), decltype(setnone)>(prog, l, listappend, setattr, setnone);
  return;
}

static void
set_instructions(goto_programt &prog, boost::python::object o)
{
  using namespace boost::python;
  // Reverse the get_instructions function: generate a list of instructiont's
  // that preserve the 'target' attribute relation.

  list pylist = extract<list>(o);
  std::vector<object> py_obj_vec;
  std::vector<goto_programt::targett> obj_it_vec;
  std::map<object, unsigned int> target_map;

  prog.instructions.clear();

  // Extract list into vector we can easily index, pushing the extracted C++
  // object into the goto_programt's instruction list. Later store a vector of
  // iterators into that list: we need the instructiont storage and it's
  // iterators to stay stable, while mapping the 'target' relation back from
  // python into C++.
  for (unsigned int i = 0; i < len(pylist); i++) {
    object item = pylist[i];
    py_obj_vec.push_back(item);
    prog.instructions.push_back(extract<goto_programt::instructiont>(item));

    // XXX -- the performance of the following may be absolutely terrible,
    // it's not clear whether there's an operator< for std::map to infer
    // anywhere here. Based on assumption that a POD comparison is done against
    // the contained python ptr.
    target_map.insert(std::make_pair(item, i));
  }

  for (auto it = prog.instructions.begin(); it != prog.instructions.end(); it++)
    obj_it_vec.push_back(it);

  // Now iterate over each pair of python/c++ instructiont objs looking at the
  // 'target' attribute. Update the corresponding 'target' field of the C++
  // object accordingly
  for (unsigned int i = 0; i < py_obj_vec.size(); i++) {
    object target = py_obj_vec[i].attr("target");
    auto it = obj_it_vec[i];

    if (target.is_none()) {
      it->targets.clear();
    } else {
      // Record a target -- map object to index, and from there to a list iter
      auto map_it = target_map.find(target);
      // Python user is entirely entitled to plug an arbitary object in here,
      // in which case we explode. Could raise an exception, but I prefer to
      // fail fast & fail hard. This isn't something the user should handle
      // anyway, and it's difficult for us to clean up afterwards.
      assert(map_it != target_map.end() && "Target PyObject of instruction is not in list");

      auto target_list_it = obj_it_vec[map_it->second];
      it->targets.clear();
      it->targets.push_back(target_list_it);
    }
  }

  return;
}

std::string
prog_to_string(const goto_programt &prog)
{
  std::stringstream ss;
  prog.output(ss);
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

  list.push_back(insn);
  goto_programt::output_instruction(namespacet(ctx), "", ss,
                                          list.begin(), show_location,
                                          show_variables);

  return ss.str();
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
    .def("is_end_function", &insnt::is_end_function);
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
    .def("empty", &goto_programt::empty);
}

#endif
