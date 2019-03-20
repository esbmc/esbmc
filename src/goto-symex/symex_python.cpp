#ifdef WITH_PYTHON
#include <sstream>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/operators.hpp>
#include <boost/python/override.hpp>
#include <util/bp_converter.h>
#include <solvers/smt_python.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/slice.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/build_goto_trace.h>

class dummy_renaming_class
{
};

// To fully implement all it's desired methods, boost's vector indexing suite
// provides a contains() method that it doesn't (appear to) use itself.
// This means that you can use the 'in' operator on the vector from python.
// However that doesn't semantically make sense (why would you search for a
// goto_symex_statet...), so I don't see a point in implementing it. Therefore
// abort if someone tries to do that from python.
// One could throw, but I think abort gets the nuances of the point across
// better.
bool operator==(const goto_symex_statet &a, const goto_symex_statet &b)
{
  (void)a;
  (void)b;
  std::cerr << "Something called contains() or similar on a boost python "
               "vector of goto_symex_statet's: don't do that."
            << std::endl;
  abort();
}

// Also: framet.
bool operator==(
  const goto_symex_statet::framet &a,
  const goto_symex_statet::framet &b)
{
  (void)a;
  (void)b;
  std::cerr << "Something called contains() or similar on a boost python "
               "vector of goto_symex_statet::framet's: don't do that."
            << std::endl;
  abort();
}

class python_rt_mangler
{
public:
  static boost::python::object get_execution_states(reachability_treet &art)
  {
    using namespace boost::python;

    list l;
    for(auto const &ex : art.execution_states)
      l.append(object(ex));

    return l;
  }

  static void
  set_execution_states(reachability_treet &art, boost::python::object &o)
  {
    using namespace boost::python;

    list pylist = extract<list>(o);
    std::list<boost::shared_ptr<execution_statet>> replacement_list;

    unsigned int i;
    for(i = 0; i < len(pylist); i++)
    {
      object elem = pylist[i];
      boost::shared_ptr<execution_statet> ex =
        extract<boost::shared_ptr<execution_statet>>(elem);
      replacement_list.push_back(ex);
    }

    art.execution_states.clear();
    art.execution_states = replacement_list;
    return;
  }

  static void set_cur_state(reachability_treet &art, execution_statet *ex_state)
  {
    // Something in the list of ex_states should have that ptr.

    decltype(art.cur_state_it) foo; // iterator on ex states
    for(foo = art.execution_states.begin(); foo != art.execution_states.end();
        foo++)
    {
      if(foo->get() == ex_state)
        break;
    }

    if(foo == art.execution_states.end())
      throw "That execution state wasn't in the list";

    art.cur_state_it = foo;
    return;
  }
};

template <typename Base>
class ex_state_wrapper : public Base, public boost::python::wrapper<Base>
{
public:
  // Template forwarding constructor.
  template <typename... Args>
  ex_state_wrapper(Args &... args) : Base(args...)
  {
  }

  void symex_step(reachability_treet &art)
  {
    using namespace boost::python;
    if(override f = this->get_override("symex_step"))
      f(art);
    else
      Base::symex_step(art);
  }

  void default_symex_step(reachability_treet &art)
  {
    Base::symex_step(art);
  }

  boost::shared_ptr<execution_statet> clone(void) const
  {
    using namespace boost::python;
    if(override f = this->get_override("clone"))
      return f();

    return Base::clone();
  }

  boost::shared_ptr<execution_statet> default_clone() const
  {
    return Base::clone();
  }
};

const value_sett &get_value_set(const goto_symex_statet &ss)
{
  return ss.value_set;
}

const renaming::level2t &get_level2(const goto_symex_statet &ss)
{
  return ss.level2;
}

const renaming::level2t &
get_state_level2(const goto_symex_statet::goto_statet &state)
{
  return state.level2;
}

static std::string goto_trace_2_text(goto_tracet &trace, const namespacet &ns)
{
  std::stringstream ss;
  show_goto_trace(ss, ns, trace);
  return ss.str();
}

static symex_targett::sourcet get_frame_source(const stack_framet &ref)
{
  if(ref.src)
    return *ref.src;

  throw "";
}

class dummy_symex_class
{
};
void build_equation_class();

template <typename Base>
class ste_wrapper : public Base, public boost::python::wrapper<Base>
{
public:
  // Template forwarding constructor.
  template <typename... Args>
  ste_wrapper(Args &... args) : Base(args...)
  {
  }

  void symex_step(reachability_treet &art)
  {
    using namespace boost::python;
    if(override f = this->get_override("symex_step"))
      f(art);
    else
      Base::symex_step(art);
  }

  void default_symex_step(reachability_treet &art)
  {
    Base::symex_step(art);
  }

  void assignment(
    const expr2tc &guard,
    const expr2tc &lhs,
    const expr2tc &original_lhs,
    const expr2tc &rhs,
    const typename Base::sourcet &source,
    std::vector<stack_framet> stack_trace,
    const bool hidden,
    unsigned loop_number)
  {
    using namespace boost::python;
    if(override f = this->get_override("assignment"))
      f(guard,
        lhs,
        original_lhs,
        rhs,
        source,
        stack_trace,
        hidden,
        loop_number);
    else
      Base::assignment(
        guard,
        lhs,
        original_lhs,
        rhs,
        source,
        stack_trace,
        hidden,
        loop_number);
  }

  void default_assignment(
    const expr2tc &guard,
    const expr2tc &lhs,
    const expr2tc &original_lhs,
    const expr2tc &rhs,
    const typename Base::sourcet &source,
    std::vector<stack_framet> stack_trace,
    const bool hidden,
    unsigned loop_number)
  {
    Base::assignment(
      guard, lhs, original_lhs, rhs, source, stack_trace, hidden, loop_number);
  }

  void assertion(
    const expr2tc &guard,
    const expr2tc &cond,
    const std::string &msg,
    std::vector<stack_framet> stack_trace,
    const typename Base::sourcet &source,
    unsigned loop_number)
  {
    using namespace boost::python;
    if(override f = this->get_override("assertion"))
      f(guard, cond, msg, stack_trace, source, loop_number);
    else
      Base::assertion(guard, cond, msg, stack_trace, source, loop_number);
  }

  void default_assertion(
    const expr2tc &guard,
    const expr2tc &cond,
    const std::string &msg,
    std::vector<stack_framet> stack_trace,
    const typename Base::sourcet &source,
    unsigned loop_number)
  {
    Base::assertion(guard, cond, msg, stack_trace, source, loop_number);
  }

  void assumption(
    const expr2tc &guard,
    const expr2tc &cond,
    const typename Base::sourcet &source,
    unsigned loop_number)
  {
    using namespace boost::python;
    if(override f = this->get_override("assumption"))
      f(guard, cond, source, loop_number);
    else
      Base::assumption(guard, cond, source, loop_number);
  }

  void default_assumption(
    const expr2tc &guard,
    const expr2tc &cond,
    const typename Base::sourcet &source,
    unsigned loop_number)
  {
    Base::assumption(guard, cond, source, loop_number);
  }

  // NB: if you override anything, but not clone, then those overrides will
  // magically disappear upon cloning, because the ste object was cloned but
  // not the containing python object.
  boost::shared_ptr<symex_targett> clone(void) const
  {
    using namespace boost::python;
    if(override f = this->get_override("clone"))
      return f();

    // The returned object is _not_ an ste_wrapper. Which is the right
    // behaviour: otherwise we'd have two c++ objs reffing the python obj
    return Base::clone();
  }

  boost::shared_ptr<symex_targett> default_clone(void) const
  {
    // See above
    return Base::clone();
  }
};

void build_goto_symex_classes()
{
  using namespace boost::python;

  scope esbmc;

  object symex(handle<>(borrowed(PyImport_AddModule("esbmc.symex"))));
  scope quux = symex;

  esbmc.attr("symex") = symex;

  symex.attr("slice") = make_function(&::slice);
  symex.attr("simple_slice") = make_function(&::simple_slice);
  symex.attr("build_goto_trace") = make_function(&build_goto_trace);

  build_equation_class();

  {
    using namespace renaming;
    object renam(
      handle<>(borrowed(PyImport_AddModule("esbmc.symex.renaming"))));
    scope corge = renam;

    symex.attr("renaming") = renam;

    void (renaming_levelt::*get_original_name)(expr2tc & expr) const =
      &renaming_levelt::get_original_name;
    class_<renaming_levelt, boost::noncopyable>("renaming_levelt", no_init)
      .def("get_original_name", get_original_name)
      .def("rename", &renaming_levelt::rename)
      .def("remove", &renaming_levelt::remove)
      .def("get_ident_name", &renaming_levelt::get_ident_name);

    {
      void (level1t::*rename)(expr2tc &) = &level1t::rename;
      void (level1t::*rename_frame)(const expr2tc &, unsigned int) =
        &level1t::rename;
      class_<level1t, bases<renaming_levelt>>("level1t")
        .def("current_number", &level1t::current_number)
        .def("get_original_name", &level1t::get_original_name)
        .def("rename", rename)
        .def("remove", &level1t::remove)
        .def("get_ident_name", &level1t::get_ident_name)
        .def("rename_frame", rename_frame)
        .def_readwrite("thread_id", &level1t::thread_id)
        .def_readwrite("current_names", &level1t::current_names);
    }

    class_<level1t::current_namest>("level1_current_names")
      .def(map_indexing_suite<level1t::current_namest, true>());

    using boost::python::self_ns::self;
    class_<level1t::name_record>("level1_name_record", init<irep_idt &>())
      .def_readwrite("base_name", &level1t::name_record::base_name)
      .def(self < self)
      .def(self == self);

    class_<level2t::valuet>("level2_value")
      .def_readwrite("count", &level2t::valuet::count)
      .def_readwrite("constant", &level2t::valuet::constant)
      .def_readwrite("node_id", &level2t::valuet::node_id);

    unsigned (level2t::*current_number)(const expr2tc &sym) const =
      &level2t::current_number;
    unsigned (level2t::*current_number_rec)(const level2t::name_record &rec)
      const = &level2t::current_number;
    void (level2t::*rename)(expr2tc &) = &level2t::rename;
    void (level2t::*rename_num)(expr2tc &, unsigned) = &level2t::rename;
    void (level2t::*remove)(const expr2tc &) = &level2t::remove;
    void (level2t::*remove_rec)(const level2t::name_record &) =
      &level2t::remove;
    class_<level2t, bases<renaming_levelt>, boost::noncopyable>(
      "level2t", no_init)
      .def("dump", &level2t::dump)
      .def("clone", &level2t::clone)
      .def("current_number", current_number)
      .def("current_number_record", current_number_rec)
      .def("get_original_name", &level2t::get_original_name)
      .def("make_assignment", &level2t::make_assignment)
      .def("rename", rename)
      .def("rename_num", rename_num)
      .def("remove", remove)
      .def("remove_record", remove_rec)
      .def_readwrite("current_names", &level2t::current_names);

    class_<level2t::name_record>("level2_name_record", init<symbol2t &>())
      .def_readwrite("base_name", &level2t::name_record::base_name)
      .def_readwrite("lev", &level2t::name_record::lev)
      .def_readwrite("l1_num", &level2t::name_record::l1_num)
      .def_readwrite("t_num", &level2t::name_record::t_num)
      .def_readwrite("hash", &level2t::name_record::hash);

    class_<level2t::current_namest>("level2_current_names")
      .def(map_indexing_suite<level2t::current_namest, true>());
  }

  // Overload resolve...
  void (goto_symext::*do_simplify)(expr2tc &) = &goto_symext::do_simplify;

  class_<goto_symext, boost::noncopyable>("goto_symex", no_init)
    .def("guard_identifier", &goto_symext::guard_identifier)
    .def("get_symex_result", &goto_symext::get_symex_result)
    .def("symex_step", &goto_symext::symex_step)
    .def("finish_formula", &goto_symext::finish_formula)
    .def("do_simplify", do_simplify)
    .def("dereference", &goto_symext::dereference)
    .def("symex_goto", &goto_symext::symex_goto)
    .def("symex_return", &goto_symext::symex_other)
    .def("claim", &goto_symext::claim)
    .def("assume", &goto_symext::assume)
    .def("merge_gotos", &goto_symext::merge_gotos)
    .def("merge_value_sets", &goto_symext::merge_value_sets)
    .def("phi_function", &goto_symext::phi_function)
    .def("get_unwind", &goto_symext::get_unwind)
    .def("loop_bound_exceeded", &goto_symext::loop_bound_exceeded)
    .def("pop_frame", &goto_symext::pop_frame)
    .def("make_return_assignment", &goto_symext::make_return_assignment)
    .def("symex_function_call", &goto_symext::symex_function_call)
    .def("symex_end_of_function", &goto_symext::symex_end_of_function)
    .def("symex_function_call_deref", &goto_symext::symex_function_call_deref)
    .def("symex_function_call_code", &goto_symext::symex_function_call_code)
    .def("get_unwind_recursion", &goto_symext::get_unwind_recursion)
    .def("argument_assignments", &goto_symext::argument_assignments)
    .def("locality", &goto_symext::locality)
    .def(
      "run_next_function_ptr_target",
      &goto_symext::run_next_function_ptr_target)
    .def("symex_realloc", &goto_symext::symex_realloc)
    .def("run_intrinsic", &goto_symext::run_intrinsic)
    .def("intrinsic_yield", &goto_symext::intrinsic_yield)
    .def("intrinsic_switch_to", &goto_symext::intrinsic_switch_to)
    .def("intrinsic_switch_from", &goto_symext::intrinsic_switch_from)
    .def("intrinsic_get_thread_id", &goto_symext::intrinsic_get_thread_id)
    .def("intrinsic_set_thread_data", &goto_symext::intrinsic_set_thread_data)
    .def("intrinsic_get_thread_data", &goto_symext::intrinsic_get_thread_data)
    .def("intrinsic_spawn_thread", &goto_symext::intrinsic_spawn_thread)
    .def("intrinsic_terminate_thread", &goto_symext::intrinsic_terminate_thread)
    .def("intrinsic_get_thread_state", &goto_symext::intrinsic_get_thread_state)
    .def(
      "intrinsic_really_atomic_begin",
      &goto_symext::intrinsic_really_atomic_begin)
    .def(
      "intrinsic_really_atomic_end", &goto_symext::intrinsic_really_atomic_end)
    .def("symex_throw", &goto_symext::symex_throw)
    .def("symex_catch", &goto_symext::symex_catch)
    .def("symex_throw_decl", &goto_symext::symex_throw_decl)
    .def("update_throw_target", &goto_symext::update_throw_target)
    .def("handle_rethrow", &goto_symext::handle_rethrow)
    .def("handle_throw_decl", &goto_symext::handle_throw_decl)
    .def("terminate_handler", &goto_symext::terminate_handler)
    .def("unexpected_handler", &goto_symext::unexpected_handler)
    .def("replace_dynamic_allocation", &goto_symext::replace_dynamic_allocation)
    .def(
      "default_replace_dynamic_allocation",
      &goto_symext::default_replace_dynamic_allocation)
    .def("is_valid_object", &goto_symext::is_valid_object)
    .def("symex_assign", &goto_symext::symex_assign)
    .def("symex_assign_rec", &goto_symext::symex_assign_rec)
    .def("symex_assign_symbol", &goto_symext::symex_assign_symbol)
    .def("symex_assign_structure", &goto_symext::symex_assign_structure)
    .def("symex_assign_typecast", &goto_symext::symex_assign_typecast)
    .def("symex_assign_array", &goto_symext::symex_assign_array)
    .def("symex_assign_member", &goto_symext::symex_assign_member)
    .def("symex_assign_if", &goto_symext::symex_assign_if)
    .def("symex_assign_byte_extract", &goto_symext::symex_assign_byte_extract)
    .def("symex_assign_concat", &goto_symext::symex_assign_concat)
    .def("symex_malloc", &goto_symext::symex_malloc)
    .def("symex_alloca", &goto_symext::symex_alloca)
    .def("symex_mem", &goto_symext::symex_mem)
    .def("track_new_pointer", &goto_symext::track_new_pointer)
    .def("symex_free", &goto_symext::symex_free)
    .def("symex_cpp_delete", &goto_symext::symex_cpp_delete)
    .def("symex_cpp_new", &goto_symext::symex_cpp_new)
    .def("replace_nondet", &goto_symext::replace_nondet)
    .def_readwrite("guard_identifier_s", &goto_symext::guard_identifier_s)
    .def_readwrite("total_claims", &goto_symext::total_claims)
    .def_readwrite("remaining_claims", &goto_symext::remaining_claims)
    .def_readwrite("art1", &goto_symext::art1)
    .def_readwrite("unwind_set", &goto_symext::unwind_set)
    .def_readwrite("max_unwind", &goto_symext::max_unwind)
    .def_readwrite("constant_propagation", &goto_symext::constant_propagation)
    .def_readwrite("target", &goto_symext::target)
    .def_readwrite("cur_state", &goto_symext::cur_state)
    .def_readwrite("valid_ptr_arr_name", &goto_symext::valid_ptr_arr_name)
    .def_readwrite("alloc_size_arr_name", &goto_symext::alloc_size_arr_name)
    .def_readwrite("deallocd_arr_name", &goto_symext::deallocd_arr_name)
    .def_readwrite("dyn_info_arr_name", &goto_symext::dyn_info_arr_name)
    .def_readwrite("dynamic_memory", &goto_symext::dynamic_memory)
    .def_readwrite("stack_catch", &goto_symext::stack_catch)
    .def_readwrite("last_throw", &goto_symext::last_throw)
    .def_readwrite("thrown_obj_map", &goto_symext::thrown_obj_map)
    .def_readwrite("inside_unexpected", &goto_symext::inside_unexpected)
    .def_readwrite("depth_limit", &goto_symext::depth_limit)
    .def_readwrite("break_insn", &goto_symext::break_insn)
    .def_readwrite("memory_leak_check", &goto_symext::memory_leak_check)
    .def_readwrite("no_assertions", &goto_symext::no_assertions)
    .def_readwrite("no_simplify", &goto_symext::no_simplify)
    .def_readwrite(
      "no_unwinding_assertions", &goto_symext::no_unwinding_assertions)
    .def_readwrite("partial_loops", &goto_symext::no_unwinding_assertions)
    .def_readwrite("body_warnings", &goto_symext::body_warnings)
    .def_readwrite("internal_deref_items", &goto_symext::internal_deref_items);

  class_<goto_symex_statet::goto_statet>(
    "goto_statet", init<goto_symex_statet>())
    .def_readwrite("depth", &goto_symex_statet::goto_statet::depth)
    .def_readwrite("value_set", &goto_symex_statet::goto_statet::value_set)
    .def_readwrite("guard", &goto_symex_statet::goto_statet::guard)
    .def_readwrite("thread_id", &goto_symex_statet::goto_statet::thread_id)
    .add_property(
      "level2", make_function(get_state_level2, return_internal_reference<>()));

  class_<goto_symex_statet::framet>("framet", init<unsigned int>())
    .def_readwrite(
      "function_identifier", &goto_symex_statet::framet::function_identifier)
    // XXX -- exporting this is going to require serious hijinx.
    .def_readwrite("goto_state_map", &goto_symex_statet::framet::goto_state_map)
    .def_readwrite("level1", &goto_symex_statet::framet::level1)
    .def_readwrite(
      "calling_location", &goto_symex_statet::framet::calling_location)
    .def_readwrite(
      "end_of_function", &goto_symex_statet::framet::end_of_function)
    .def_readwrite("return_value", &goto_symex_statet::framet::return_value)
    .def_readwrite(
      "local_variables", &goto_symex_statet::framet::local_variables)
    .def_readwrite(
      "cur_function_ptr_targets",
      &goto_symex_statet::framet::cur_function_ptr_targets)
    .def_readwrite(
      "function_ptr_call_loc",
      &goto_symex_statet::framet::function_ptr_call_loc)
    .def_readwrite(
      "function_ptr_combine_target",
      &goto_symex_statet::framet::function_ptr_combine_target)
    .def_readwrite(
      "orig_func_ptr_call", &goto_symex_statet::framet::orig_func_ptr_call)
    .def_readwrite(
      "declaration_history", &goto_symex_statet::framet::declaration_history);

  class_<goto_symex_statet::loop_iterationst>("loop_iterationst")
    .def(map_indexing_suite<hash_map_cont<unsigned, unsigned>>());

  void (goto_symex_statet::*current_name_expr)(expr2tc &) const =
    &goto_symex_statet::current_name;
  void (goto_symex_statet::*current_name_level)(
    const renaming::level2t &, expr2tc &) const =
    &goto_symex_statet::current_name;
  void (goto_symex_statet::*current_name_state)(
    const goto_symex_statet::goto_statet &, expr2tc &) const =
    &goto_symex_statet::current_name;
  goto_symex_statet::framet &(goto_symex_statet::*top)(void) =
    &goto_symex_statet::top;

  class_<goto_symex_statet, boost::noncopyable>(
    "goto_symex_statet",
    init<renaming::level2t &, value_sett &, namespacet &>())
    .def("current_name_expr", current_name_expr)
    .def("current_name_level", current_name_level)
    .def("current_name_state", current_name_state)
    .def("top", top, return_internal_reference<>())
    .def(
      "new_frame", &goto_symex_statet::new_frame, return_internal_reference<>())
    .def("pop_frame", &goto_symex_statet::pop_frame)
    .def(
      "previous_frame",
      &goto_symex_statet::previous_frame,
      return_internal_reference<>())
    .def("initialize", &goto_symex_statet::initialize)
    .def("rename", &goto_symex_statet::rename)
    .def("rename_address", &goto_symex_statet::rename_address)
    .def("assignment", &goto_symex_statet::assignment)
    .def("constant_propagation", &goto_symex_statet::constant_propagation)
    .def(
      "constant_propagation_reference",
      &goto_symex_statet::constant_propagation_reference)
    .def("get_original_name", &goto_symex_statet::get_original_name)
    .def("print_stack_trace", &goto_symex_statet::print_stack_trace)
    .def("gen_stack_trace", &goto_symex_statet::gen_stack_trace)
    .def("fixup_renamed_type", &goto_symex_statet::fixup_renamed_type)
    .def_readwrite("depth", &goto_symex_statet::depth)
    .def_readwrite("thread_ended", &goto_symex_statet::thread_ended)
    .def_readwrite("guard", &goto_symex_statet::guard)
    .def_readwrite("global_guard", &goto_symex_statet::global_guard)
    .def_readwrite("source", &goto_symex_statet::source)
    .def_readwrite(
      "variable_instance_nums", &goto_symex_statet::variable_instance_nums)
    .def_readwrite("function_unwind", &goto_symex_statet::function_unwind)
    .def_readwrite("use_value_set", &goto_symex_statet::use_value_set)
    .add_property(
      "level2", make_function(get_level2, return_internal_reference<>()))
    .add_property(
      "value_set", make_function(get_value_set, return_internal_reference<>()))
    .def_readwrite("call_stack", &goto_symex_statet::call_stack)
    .def_readwrite("loop_iterations", &goto_symex_statet::loop_iterations)
    .def_readwrite("realloc_map", &goto_symex_statet::realloc_map);

  class_<std::vector<goto_symex_statet::framet>>("frame_vec")
    .def(vector_indexing_suite<std::vector<goto_symex_statet::framet>>());

  {
    // Resolve overloads:
    goto_symex_statet &(execution_statet::*get_active_state)() =
      &execution_statet::get_active_state;

    scope fgasdf =
      class_<
        execution_statet,
        boost::noncopyable,
        bases<goto_symext>,
        boost::shared_ptr<execution_statet>>(
        "execution_state", no_init) // is abstract
        .def(
          "increment_context_switch",
          &execution_statet::increment_context_switch)
        .def("increment_time_slice", &execution_statet::increment_time_slice)
        .def("reset_time_slice", &execution_statet::reset_time_slice)
        .def("get_context_switch", &execution_statet::get_context_switch)
        .def("get_time_slice", &execution_statet::get_time_slice)
        .def("resetDFS_traversed", &execution_statet::resetDFS_traversed)
        .def(
          "get_active_state_number", &execution_statet::get_active_state_number)
        .def("set_thread_start_data", &execution_statet::set_thread_start_data)
        .def(
          "get_thread_start_data",
          &execution_statet::get_thread_start_data,
          return_internal_reference<>())
        .def("clone", &execution_statet::clone)
        .def("symex_step", &execution_statet::symex_step)
        .def("symex_assign", &execution_statet::symex_assign)
        .def("claim", &execution_statet::claim)
        .def("symex_goto", &execution_statet::symex_goto)
        .def("assume", &execution_statet::assume)
#if 0
    // boost.python won't wrap these
    .def("get_dynamic_counter", &execution_statet::get_dynamic_counter, return_internal_reference<>())
    .def("get_nondet_counter", &execution_statet::get_nondet_counter, return_internal_reference<>())
#endif
        .def("get_guard_identifier", &execution_statet::get_guard_identifier)
        .def(
          "get_active_state", get_active_state, return_internal_reference<>())
        .def(
          "get_active_atomic_number",
          &execution_statet::get_active_atomic_number)
        .def(
          "increment_active_atomic_number",
          &execution_statet::increment_active_atomic_number)
        .def(
          "decrement_active_atomic_number",
          &execution_statet::decrement_active_atomic_number)
        .def("switch_to_thread", &execution_statet::switch_to_thread)
        .def(
          "is_cur_state_guard_false",
          &execution_statet::is_cur_state_guard_false)
        .def("execute_guard", &execution_statet::execute_guard)
        .def("dfs_explore_thread", &execution_statet::dfs_explore_thread)
        .def(
          "check_if_ileaves_blocked",
          &execution_statet::check_if_ileaves_blocked)
        .def("add_thread", &execution_statet::add_thread)
        .def("end_thread", &execution_statet::end_thread)
        .def(
          "update_after_switch_point",
          &execution_statet::update_after_switch_point)
        .def("analyze_assign", &execution_statet::analyze_assign)
        .def("analyze_read", &execution_statet::analyze_read)
        .def("get_expr_globals", &execution_statet::get_expr_globals)
        .def("check_mpor_dependancy", &execution_statet::check_mpor_dependancy)
        .def(
          "calculate_mpor_constraints",
          &execution_statet::calculate_mpor_constraints)
        .def(
          "is_transition_blocked_by_mpor",
          &execution_statet::is_transition_blocked_by_mpor)
        .def("force_cswitch", &execution_statet::force_cswitch)
        .def(
          "has_cswitch_point_occured",
          &execution_statet::has_cswitch_point_occured)
        .def(
          "can_execution_continue", &execution_statet::can_execution_continue)
        .def("print_stack_traces", &execution_statet::print_stack_traces)
        .def_readwrite("owning_rt", &execution_statet::owning_rt)
        .def_readwrite("threads_state", &execution_statet::threads_state)
        .def_readwrite("atomic_numbers", &execution_statet::atomic_numbers)
        .def_readwrite("DFS_traversed", &execution_statet::DFS_traversed)
        .def_readwrite(
          "thread_start_data", &execution_statet::thread_start_data)
        .def_readwrite(
          "last_active_thread", &execution_statet::last_active_thread)
        .def_readwrite("state_level2", &execution_statet::state_level2)
        .def_readwrite(
          "global_value_set", &execution_statet::global_value_set) // :O:O:O:O
        .def_readwrite("active_thread", &execution_statet::active_thread)
        .def_readwrite("guard_execution", &execution_statet::guard_execution)
        .def_readwrite("TS_number", &execution_statet::TS_number)
        .def_readwrite("nondet_count", &execution_statet::nondet_count)
        .def_readwrite("dynamic_counter", &execution_statet::dynamic_counter)
        .def_readwrite("node_id", &execution_statet::node_id)
        .def_readwrite(
          "interleaving_unviable", &execution_statet::interleaving_unviable)
        .def_readwrite("pre_goto_guard", &execution_statet::pre_goto_guard)
        .def_readwrite("CS_number", &execution_statet::CS_number)
        // These two fields can't currently be accessed as it's a set, _inside_ a
        // vector. Converting the internal sets to python means we can't use the
        // vector indexing suite and vice versa. No good solution right now, and
        // it's too much investment for too little return.
        .def_readwrite(
          "thread_last_reads", &execution_statet::thread_last_reads)
        .def_readwrite(
          "thread_last_writes", &execution_statet::thread_last_writes)

        .def_readwrite("dependancy_chain", &execution_statet::dependancy_chain)
        .def_readwrite("mpor_says_no", &execution_statet::mpor_says_no)
        .def_readwrite("cswitch_forced", &execution_statet::cswitch_forced)
        .def_readwrite("symex_trace", &execution_statet::symex_trace)
        .def_readwrite("smt_during_symex", &execution_statet::smt_during_symex)
        .def_readwrite("smt_thread_guard", &execution_statet::smt_thread_guard)
        .def_readwrite("node_count", &execution_statet::node_count);

    // NB: this requires an operator== to be defined for goto_symex_statet,
    // see above.
    class_<std::vector<goto_symex_statet>>("goto_symex_statet_vec")
      .def(vector_indexing_suite<std::vector<goto_symex_statet>>());
    class_<std::vector<unsigned int>>("atomic_nums_vec")
      .def(vector_indexing_suite<std::vector<unsigned>>());
    class_<std::vector<bool>>("dfs_state_vec")
      .def(vector_indexing_suite<std::vector<bool>>());

    // Resolve overload...
    void (execution_statet::ex_state_level2t::*rename_to)(
      expr2tc & lhs_symbol, unsigned count) =
      &execution_statet::ex_state_level2t::rename;
    void (execution_statet::ex_state_level2t::*rename)(expr2tc & ident) =
      &execution_statet::ex_state_level2t::rename;

    // xxx base
    class_<
      execution_statet::ex_state_level2t,
      boost::shared_ptr<execution_statet::ex_state_level2t>>(
      "ex_state_level2t", init<execution_statet &>())
      .def("clone", &execution_statet::ex_state_level2t::clone)
      .def("rename_to", rename_to)
      .def("rename", rename)
      .def_readwrite("owner", &execution_statet::ex_state_level2t::owner);

    // Classes we can actually construct...
    class_<
      ex_state_wrapper<dfs_execution_statet>,
      bases<execution_statet>,
      boost::shared_ptr<ex_state_wrapper<dfs_execution_statet>>>(
      "dfs_execution_state",
      init<
        const goto_functionst &,
        const namespacet &,
        reachability_treet *,
        boost::shared_ptr<symex_targett>,
        contextt &,
        optionst &,
        message_handlert &>())
      .def(
        "symex_step",
        &goto_symext::symex_step,
        &ex_state_wrapper<dfs_execution_statet>::default_symex_step)
      .def(
        "clone",
        &dfs_execution_statet::clone,
        &ex_state_wrapper<dfs_execution_statet>::default_clone);

    class_<
      ex_state_wrapper<schedule_execution_statet>,
      bases<execution_statet>,
      boost::shared_ptr<ex_state_wrapper<schedule_execution_statet>>>(
      "schedule_execution_state",
      init<
        const goto_functionst &,
        const namespacet &,
        reachability_treet *,
        boost::shared_ptr<symex_targett>,
        contextt &,
        optionst &,
        unsigned int *,
        unsigned int *,
        message_handlert &>())
      .def(
        "symex_step",
        &goto_symext::symex_step,
        &ex_state_wrapper<schedule_execution_statet>::default_symex_step)
      .def(
        "clone",
        &schedule_execution_statet::clone,
        &ex_state_wrapper<schedule_execution_statet>::clone);

  } // ex_state scope

  // Resolve some overloads
  execution_statet &(reachability_treet::*get_cur_state)() =
    &reachability_treet::get_cur_state;
  class_<reachability_treet>(
    "reachability_tree",
    init<
      goto_functionst &,
      namespacet &,
      optionst &,
      boost::shared_ptr<symex_targett>,
      contextt &,
      message_handlert &>())
    .def("setup_for_new_explore", &reachability_treet::setup_for_new_explore)
    .def("get_cur_state", get_cur_state, return_internal_reference<>())
    .def("set_cur_state", &python_rt_mangler::set_cur_state)
    .def(
      "reset_to_unexplored_state",
      &reachability_treet::reset_to_unexplored_state)
    .def("get_CS_bound", &reachability_treet::get_CS_bound)
    .def(
      "get_ileave_direction_from_scheduling",
      &reachability_treet::get_ileave_direction_from_scheduling)
    .def("check_thread_viable", &reachability_treet::check_thread_viable)
    .def("create_next_state", &reachability_treet::create_next_state)
    .def("step_next_state", &reachability_treet::step_next_state)
    .def(
      "decide_ileave_direction", &reachability_treet::decide_ileave_direction)
    .def("print_ileave_trace", &reachability_treet::print_ileave_trace)
    .def(
      "is_has_complete_formula", &reachability_treet::is_has_complete_formula)
    .def("go_next_state", &reachability_treet::go_next_state)
    .def(
      "switch_to_next_execution_state",
      &reachability_treet::switch_to_next_execution_state)
    .def("get_next_formula", &reachability_treet::get_next_formula)
    .def(
      "generate_schedule_formula",
      &reachability_treet::generate_schedule_formula)
    .def("setup_next_formula", &reachability_treet::setup_next_formula)
    .def_readwrite(
      "has_complete_formula", &reachability_treet::has_complete_formula)
    .add_property(
      "execution_states",
      make_function(&python_rt_mangler::get_execution_states),
      make_function(&python_rt_mangler::set_execution_states))
    .def_readwrite("cur_state_it", &reachability_treet::cur_state_it)
    .def_readwrite("schedule_target", &reachability_treet::schedule_target)
    .def_readwrite("target_template", &reachability_treet::target_template)
    .def_readwrite("CS_bound", &reachability_treet::CS_bound)
    .def_readwrite("TS_slice", &reachability_treet::TS_slice)
    .def_readwrite(
      "schedule_total_claims", &reachability_treet::schedule_total_claims)
    .def_readwrite(
      "schedule_remaining_claims",
      &reachability_treet::schedule_remaining_claims)
    .def_readwrite("next_thread_id", &reachability_treet::next_thread_id)
    .def_readwrite("por", &reachability_treet::por)
    .def_readwrite("round_robin", &reachability_treet::round_robin)
    .def_readwrite("schedule", &reachability_treet::schedule);

  return;
}

static boost::python::object
get_guard_ast(symex_target_equationt::SSA_stept &step)
{
  return smt_ast_wrapper::cast_ast_down(step.guard_ast);
}

static void
set_guard_ast(symex_target_equationt::SSA_stept &step, const smt_ast *ast)
{
  step.guard_ast = ast;
  return;
}

static boost::python::object
get_cond_ast(symex_target_equationt::SSA_stept &step)
{
  return smt_ast_wrapper::cast_ast_down(step.cond_ast);
}

static void
set_cond_ast(symex_target_equationt::SSA_stept &step, const smt_ast *ast)
{
  step.cond_ast = ast;
  return;
}

static boost::python::object get_insns(symex_target_equationt *eq)
{
  using namespace boost::python;

  list l;

  for(auto const &step : eq->SSA_steps)
    l.append(object(&step));

  return l;
}

static void
append_insn(symex_target_equationt *eq, symex_target_equationt::SSA_stept &step)
{
  eq->SSA_steps.push_back(step);
  return;
}

static void pop_insn(symex_target_equationt *eq)
{
  // Don't return anything, has no storage
  eq->SSA_steps.pop_back();
  return;
}

static symex_target_equationt::SSA_stept peek_insn(symex_target_equationt *eq)
{
  return eq->SSA_steps.back();
}

static unsigned int get_pc(const symex_targett::sourcet &src)
{
  return src.pc->location_number;
}

static void set_pc(symex_targett::sourcet &src, unsigned int loc)
{
  if(!src.prog)
    throw "No program in sourcet";

  if(src.prog->instructions.size() == 0)
    throw "Empty program";

  auto &insns = src.prog->instructions;
  if(
    loc < insns.begin()->location_number ||
    insns.begin()->location_number + insns.size() < loc)
    throw "Location number not in range";

  unsigned int dist = loc - insns.begin()->location_number;
  auto it = insns.begin();
  while(dist-- > 0)
    it++;

  src.pc = it;
  return;
}

static goto_programt &get_prog(const symex_targett::sourcet &src)
{
  // Ditching const qualification because I'm not certain b.p is going to cope
  // with that, and it won't honour it anyway.
  return const_cast<goto_programt &>(*src.prog);
}

static void set_prog(symex_targett::sourcet &src, goto_programt &prog)
{
  src.prog = &prog;
  return;
}

static boost::python::object get_goto_trace_steps(const goto_tracet &trace)
{
  using namespace boost::python;

  list l;
  for(auto const &step : trace.steps)
  {
    l.append(object(step));
  }

  return l;
}

static bool cmp_stack_frame_vec(
  const std::vector<stack_framet> &a,
  const std::vector<stack_framet> &b)
{
  // XXX Can't remember how to take pointer of real operator==
  return a == b;
}

static bool ncmp_stack_frame_vec(
  const std::vector<stack_framet> &a,
  const std::vector<stack_framet> &b)
{
  // XXX Can't remember how to take pointer of real operator==
  return a != b;
}

static bool ncmp_stack_frame(const stack_framet &a, const stack_framet &b)
{
  // XXX Can't remember how to take pointer of real operator==
  return !(a == b);
}

void build_equation_class()
{
  using namespace boost::python;

  // In theory, we could filter these depending on the respective types of
  // different ssa steps being converted to python, but it seems pointless
  // to expose that to python without also replicating it in C++.
  typedef symex_target_equationt::SSA_stept step;
  class_<step>("ssa_step")
    .def_readwrite("source", &step::source)
    .def_readwrite("type", &step::type)
    .def_readwrite("stack_trace", &step::stack_trace)
    .def_readwrite("guard", &step::guard)
    .def_readwrite("lhs", &step::lhs)
    .def_readwrite("rhs", &step::rhs)
    .def_readwrite("original_lhs", &step::original_lhs)
    .def_readwrite("cond", &step::cond)
    .def_readwrite("comment", &step::comment)
    // For some reason, def_readwrite can't synthesize it's own setter
    // due to const perhaps, or smt_astt being opaque
    // Need getters because the ast needs to be downcasted before returning
    .add_property(
      "guard_ast", make_function(&get_guard_ast), make_function(&set_guard_ast))
    .add_property(
      "cond_ast", make_function(&get_cond_ast), make_function(&set_cond_ast))
    .def_readwrite("ignore", &step::ignore);

  {
    scope bar = class_<
      symex_targett,
      boost::shared_ptr<symex_targett>,
      boost::noncopyable>("symex_targett", no_init);

    class_<symex_targett::sourcet>("targett")
      .def_readwrite("is_set", &symex_targett::sourcet::is_set)
      .def_readwrite("thread_nr", &symex_targett::sourcet::thread_nr)
      // Access to the program is sketchy, but permissable. The program _should_
      // be one loaded into the goto_functionst function map to ensure that it
      // lives long enough.
      .add_property(
        "prog",
        make_function(get_prog, return_internal_reference<>()),
        make_function(set_prog))
      // Accessing the program counter is sketchy too as it's actually an
      // iterator. So, operate only on the integer identifier (i.e. the
      // location number), which is a) easily validatable and b) accessible
      // and c) can be used trivially by the python user to access the relevant
      // instruction. This means you can't fiddle with insns while symex is on
      // the fly, but that's a feature.
      .add_property("pc", make_function(get_pc), make_function(set_pc));
  }

  class_<
    goto_symext::symex_resultt,
    boost::shared_ptr<goto_symext::symex_resultt>>(
    "symex_resultt",
    init<boost::shared_ptr<symex_targett>, unsigned, unsigned>())
    .def_readwrite("target", &goto_symext::symex_resultt::target)
    .def_readwrite("total_claims", &goto_symext::symex_resultt::total_claims)
    .def_readwrite(
      "remaining_claims", &goto_symext::symex_resultt::remaining_claims);

  init<const namespacet &> eq_init;
  init<const ste_wrapper<symex_target_equationt> &> cpy_init;
  class_<
    ste_wrapper<symex_target_equationt>,
    boost::shared_ptr<ste_wrapper<symex_target_equationt>>,
    bases<symex_targett>>("equation", eq_init)
    .def(cpy_init)
    .def(
      "assignment",
      &symex_target_equationt::assignment,
      &ste_wrapper<symex_target_equationt>::default_assignment)
    .def(
      "assumption",
      &symex_target_equationt::assumption,
      &ste_wrapper<symex_target_equationt>::default_assumption)
    .def(
      "assertion",
      &symex_target_equationt::assertion,
      &ste_wrapper<symex_target_equationt>::default_assertion)
    .def("renumber", &symex_target_equationt::renumber)
    .def("convert", &symex_target_equationt::convert)
    .def("clear", &symex_target_equationt::clear)
    .def("check_for_dups", &symex_target_equationt::check_for_duplicate_assigns)
    .def(
      "clone",
      &symex_target_equationt::clone,
      &ste_wrapper<symex_target_equationt>::default_clone)
    .def("clear_assertions", &symex_target_equationt::clear_assertions)
    // It's way too sketchy to give python direct access to SSA_steps: we're not
    // going to convert all steps by value to python objects, and
    // self-assignment is likely to be a trainwreck. So: return a list of
    // internal references that can be messed with, and supply a facility for
    // copy-appending a step into the list or popping an insn. If you need to
    // really delicately manipulate the equation, you need to construct a new
    // one and filter insns from the old to the new.
    .def("get_insns", &get_insns)
    .def("append_insn", &append_insn)
    .def("pop_back", &pop_insn)
    .def("peek_back", &peek_insn);

  enum_<goto_trace_stept::typet>("symex_step_type")
    .value("ASSIGN", goto_trace_stept::ASSIGNMENT)
    .value("ASSUME", goto_trace_stept::ASSUME)
    .value("ASSERT", goto_trace_stept::ASSERT)
    .value("OUTPUT", goto_trace_stept::OUTPUT)
    .value("SKIP", goto_trace_stept::SKIP)
    .value("RENUMBER", goto_trace_stept::RENUMBER);

  class_<goto_trace_stept>("goto_trace_stept")
    .def_readwrite("stack_trace", &goto_trace_stept::stack_trace)
    .def_readwrite("type", &goto_trace_stept::type)
    .def_readwrite("thread_nr", &goto_trace_stept::thread_nr)
    .def_readwrite("guard", &goto_trace_stept::guard)
    .def_readwrite("comment", &goto_trace_stept::comment)
    .def_readwrite("lhs", &goto_trace_stept::lhs)
    .def_readwrite("rhs", &goto_trace_stept::rhs)
    .def_readwrite("value", &goto_trace_stept::value)
    .def_readwrite("original_lhs", &goto_trace_stept::original_lhs);

  class_<goto_tracet>("goto_tracet")
    .def("to_string", &goto_trace_2_text)
    .def("clear", &goto_tracet::clear)
    .def_readwrite("mode", &goto_tracet::mode)
    .add_property("steps", make_function(&get_goto_trace_steps));

  bool (*cmp_stack_frame)(const stack_framet &, const stack_framet &) =
    &operator==;
  class_<stack_framet>("stack_framet", no_init)
    .def_readonly("function", &stack_framet::function)
    .add_property("src", make_function(get_frame_source))
    .def("__eq__", cmp_stack_frame)
    .def("__ne__", &ncmp_stack_frame);
  class_<std::vector<stack_framet>>("stack_frame_vec")
    .def(vector_indexing_suite<std::vector<stack_framet>>())
    .def("__eq__", cmp_stack_frame_vec)
    .def("__ne__", ncmp_stack_frame_vec);
}

// A function for trapping to python. Assumptions: it's in the context of
// symbolic execution, because that's the primary purpose / function of esbmc,
// and thus an RT object must be provided. Enters the interative interpreter
// with art and other objects ready to be accessed.
extern "C" void initesbmc();
extern "C" void PyInit_esbmc();
void trap_to_python(reachability_treet *art)
{
  using namespace boost::python;

  // Check if python is initialized, set if up if not. Never de-initialize.
  if(!Py_IsInitialized())
  {
    Py_InitializeEx(0);
#if PY_VERSION_HEX >= 0x03000000
    PyInit_esbmc();
#else
    initesbmc();
#endif
  }

  object code = import("code");
  if(code.is_none())
  {
    std::cerr << "Couldn't import code module when trapping to python"
              << std::endl;
    return;
  }

  object esbmc = import("esbmc");
  if(esbmc.is_none())
  {
    std::cerr << "Couldn't import esbmc module when trapping to python"
              << std::endl;
    return;
  }

  dict locals;
  locals["art"] = object(art);
  locals["ns"] = object(art->ns);
  locals["options"] = object(art->options);
  locals["funcs"] = object(art->goto_functions);

  object interact = code.attr("interact");
  // Call interact
  interact(object(), object(), locals);

  return;
}
#endif
