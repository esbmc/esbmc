#ifdef WITH_PYTHON
#include <boost/python/class.hpp>
#include <boost/python/init.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/return_internal_reference.hpp>

#include <util/bp_opaque_ptr.h>
#include <util/bp_converter.h>

#include "reachability_tree.h"
#include "execution_state.h"
#include "goto_symex.h"

void
build_goto_symex_classes()
{
  using namespace boost::python;

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
    .def("run_next_function_ptr_target", &goto_symext::run_next_function_ptr_target)
    .def("run_intrinsic", &goto_symext::run_intrinsic)
    .def("intrinsic_realloc", &goto_symext::intrinsic_realloc)
    .def("intrinsic_yield", &goto_symext::intrinsic_yield)
    .def("intrinsic_switch_to", &goto_symext::intrinsic_switch_to)
    .def("intrinsic_switch_from", &goto_symext::intrinsic_switch_from)
    .def("intrinsic_get_thread_id", &goto_symext::intrinsic_get_thread_id)
    .def("intrinsic_set_thread_data", &goto_symext::intrinsic_set_thread_data)
    .def("intrinsic_get_thread_data", &goto_symext::intrinsic_get_thread_data)
    .def("intrinsic_spawn_thread", &goto_symext::intrinsic_spawn_thread)
    .def("intrinsic_terminate_thread", &goto_symext::intrinsic_terminate_thread)
    .def("intrinsic_get_thread_state", &goto_symext::intrinsic_get_thread_state)
    .def("intrinsic_really_atomic_begin", &goto_symext::intrinsic_really_atomic_begin)
    .def("intrinsic_really_atomic_end", &goto_symext::intrinsic_really_atomic_end)
    .def("symex_throw", &goto_symext::symex_throw)
    .def("symex_catch", &goto_symext::symex_catch)
    .def("symex_throw_decl", &goto_symext::symex_throw_decl)
    .def("update_throw_target", &goto_symext::update_throw_target)
    .def("handle_rethrow", &goto_symext::handle_rethrow)
    .def("handle_throw_decl", &goto_symext::handle_throw_decl)
    .def("terminate_handler", &goto_symext::terminate_handler)
    .def("unexpected_handler", &goto_symext::unexpected_handler)
    .def("replace_dynamic_allocation", &goto_symext::replace_dynamic_allocation)
    .def("default_replace_dynamic_allocation", &goto_symext::default_replace_dynamic_allocation)
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
    .def_readwrite("unwinding_recursion_assumption", &goto_symext::unwinding_recursion_assumption)
    .def_readwrite("depth_limit", &goto_symext::depth_limit)
    .def_readwrite("break_insn", &goto_symext::break_insn)
    .def_readwrite("memory_leak_check", &goto_symext::memory_leak_check)
    .def_readwrite("no_assertions", &goto_symext::no_assertions)
    .def_readwrite("no_simplify", &goto_symext::no_simplify)
    .def_readwrite("no_unwinding_assertions", &goto_symext::no_unwinding_assertions)
    .def_readwrite("partial_loops", &goto_symext::no_unwinding_assertions)
    .def_readwrite("body_warnings", &goto_symext::body_warnings)
    .def_readwrite("internal_deref_items", &goto_symext::internal_deref_items);

 {

  // Resolve overloads:
  goto_symex_statet& (execution_statet::*get_active_state)() =
    &execution_statet::get_active_state;

  scope fgasdf =
  class_<execution_statet, boost::noncopyable, bases<goto_symext> >("execution_state", no_init) // is abstract
    .def("increment_context_switch", &execution_statet::increment_context_switch)
    .def("increment_time_slice", &execution_statet::increment_time_slice)
    .def("reset_time_slice", &execution_statet::reset_time_slice)
    .def("get_context_switch", &execution_statet::get_context_switch)
    .def("get_time_slice", &execution_statet::get_time_slice)
    .def("resetDFS_traversed", &execution_statet::resetDFS_traversed)
    .def("get_active_state_number", &execution_statet::get_active_state_number)
    .def("set_thread_start_data", &execution_statet::set_thread_start_data)
    .def("get_thread_start_data", &execution_statet::get_thread_start_data, return_internal_reference<>())
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
    .def("get_active_state", get_active_state, return_internal_reference<>())
    .def("get_active_atomic_number", &execution_statet::get_active_atomic_number)
    .def("increment_active_atomic_number", &execution_statet::increment_active_atomic_number)
    .def("decrement_active_atomic_number", &execution_statet::decrement_active_atomic_number)
    .def("switch_to_thread", &execution_statet::switch_to_thread)
    .def("is_cur_state_guard_false", &execution_statet::is_cur_state_guard_false)
    .def("execute_guard", &execution_statet::execute_guard)
    .def("dfs_explore_thread", &execution_statet::dfs_explore_thread)
    .def("check_if_ileaves_blocked", &execution_statet::check_if_ileaves_blocked)
    .def("add_thread", &execution_statet::add_thread)
    .def("end_thread", &execution_statet::end_thread)
    .def("update_after_switch_point", &execution_statet::update_after_switch_point)
    .def("analyze_assign", &execution_statet::analyze_assign)
    .def("analyze_read", &execution_statet::analyze_read)
    .def("get_expr_globals", &execution_statet::get_expr_globals)
    .def("check_mpor_dependancy", &execution_statet::check_mpor_dependancy)
    .def("calculate_mpor_constraints", &execution_statet::calculate_mpor_constraints)
    .def("is_transition_blocked_by_mpor", &execution_statet::is_transition_blocked_by_mpor)
    .def("force_cswitch", &execution_statet::force_cswitch)
    .def("has_cswitch_point_occured", &execution_statet::has_cswitch_point_occured)
    .def("can_execution_continue", &execution_statet::can_execution_continue)
    .def("print_stack_traces", &execution_statet::print_stack_traces)
    .def_readwrite("owning_rt", &execution_statet::owning_rt)
    .def_readwrite("threads_state", &execution_statet::threads_state)
    .def_readwrite("atomic_numbers", &execution_statet::atomic_numbers)
    .def_readwrite("DFS_traversed", &execution_statet::DFS_traversed)
    .def_readwrite("thread_start_data", &execution_statet::thread_start_data)
    .def_readwrite("last_active_thread", &execution_statet::last_active_thread)
    .def_readwrite("state_level2", &execution_statet::state_level2)
    .def_readwrite("global_value_set", &execution_statet::global_value_set) // :O:O:O:O
    .def_readwrite("active_thread", &execution_statet::active_thread)
    .def_readwrite("guard_execution", &execution_statet::guard_execution)
    .def_readwrite("TS_number", &execution_statet::TS_number)
    .def_readwrite("nondet_count", &execution_statet::nondet_count)
    .def_readwrite("dynamic_counter", &execution_statet::dynamic_counter)
    .def_readwrite("node_id", &execution_statet::node_id)
    .def_readwrite("interleaving_unviable", &execution_statet::interleaving_unviable)
    .def_readwrite("pre_goto_guard", &execution_statet::pre_goto_guard)
    .def_readwrite("CS_number", &execution_statet::CS_number)
    .def_readwrite("thread_last_reads", &execution_statet::thread_last_reads)
    .def_readwrite("thread_last_writes", &execution_statet::thread_last_writes)
    .def_readwrite("dependancy_chain", &execution_statet::dependancy_chain)
    .def_readwrite("mpor_says_no", &execution_statet::mpor_says_no)
    .def_readwrite("cswitch_forced", &execution_statet::cswitch_forced)
    .def_readwrite("symex_trace", &execution_statet::symex_trace)
    .def_readwrite("smt_during_symex", &execution_statet::smt_during_symex)
    .def_readwrite("smt_thread_guard", &execution_statet::smt_thread_guard)
    .def_readwrite("node_count", &execution_statet::node_count);

  // Resolve overload...
  void (execution_statet::ex_state_level2t::*rename_to)(expr2tc &lhs_symbol, unsigned count)
    = &execution_statet::ex_state_level2t::rename;
  void (execution_statet::ex_state_level2t::*rename)(expr2tc &ident) = &execution_statet::ex_state_level2t::rename;

  // xxx base
  class_<execution_statet::ex_state_level2t>("ex_state_level2t",
      init<execution_statet&>())
    .def("clone", &execution_statet::ex_state_level2t::clone)
    .def("rename_to", rename_to)
    .def("rename", rename)
    .def_readwrite("owner", &execution_statet::ex_state_level2t::owner);

  // Classes we can actually construct...
  class_<dfs_execution_statet, bases<execution_statet> >("dfs_execution_state", 
      init<const goto_functionst &, const namespacet &, reachability_treet *,
      boost::shared_ptr<symex_targett>, contextt &,
      optionst &, message_handlert &>());

  class_<schedule_execution_statet, bases<execution_statet> >("schedule_execution_state", 
      init<const goto_functionst &, const namespacet &, reachability_treet *,
      boost::shared_ptr<symex_targett>, contextt &,
      optionst &, unsigned int *, unsigned int *, message_handlert &>());

  } // ex_state scope

  // Resolve some overloads
  execution_statet & (reachability_treet::*get_cur_state)() = &reachability_treet::get_cur_state;
  class_<reachability_treet>("reachability_tree",
    init<goto_functionst &, namespacet &, optionst &,
      boost::shared_ptr<symex_targett>, contextt &, message_handlert &>())
    .def("setup_for_new_explore", &reachability_treet::setup_for_new_explore)
    .def("get_cur_state", get_cur_state, return_internal_reference<>())
    .def("reset_to_unexplored_state", &reachability_treet::reset_to_unexplored_state)
    .def("get_CS_bound", &reachability_treet::get_CS_bound)
    .def("get_ileave_direction_from_scheduling", &reachability_treet::get_ileave_direction_from_scheduling)
    .def("check_thread_viable", &reachability_treet::check_thread_viable)
    .def("create_next_state", &reachability_treet::create_next_state)
    .def("step_next_state", &reachability_treet::step_next_state)
    .def("decide_ileave_direction", &reachability_treet::decide_ileave_direction)
    .def("print_ileave_trace", &reachability_treet::print_ileave_trace)
    .def("is_has_complete_formula", &reachability_treet::is_has_complete_formula)
    .def("go_next_state", &reachability_treet::go_next_state)
    .def("switch_to_next_execution_state", &reachability_treet::switch_to_next_execution_state)
    .def("get_next_formula", &reachability_treet::get_next_formula)
    .def("generate_schedule_formula", &reachability_treet::generate_schedule_formula)
    .def("setup_next_formula", &reachability_treet::setup_next_formula)
    .def_readwrite("has_complete_formula", &reachability_treet::has_complete_formula)
    .def_readwrite("execution_states", &reachability_treet::execution_states)
    .def_readwrite("cur_state_it", &reachability_treet::cur_state_it)
    .def_readwrite("schedule_target", &reachability_treet::schedule_target)
    .def_readwrite("target_template", &reachability_treet::target_template)
    .def_readwrite("CS_bound", &reachability_treet::CS_bound)
    .def_readwrite("TS_slice", &reachability_treet::TS_slice)
    .def_readwrite("schedule_total_claims", &reachability_treet::schedule_total_claims)
    .def_readwrite("schedule_remaining_claims", &reachability_treet::schedule_remaining_claims)
    .def_readwrite("next_thread_id", &reachability_treet::next_thread_id)
    .def_readwrite("por", &reachability_treet::por)
    .def_readwrite("round_robin", &reachability_treet::round_robin)
    .def_readwrite("schedule", &reachability_treet::schedule);

  return;
}

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_ast)

static void
set_guard_ast(symex_target_equationt::SSA_stept &step, const smt_ast *ast)
{
  step.guard_ast = ast;
  return;
}

static void
set_cond_ast(symex_target_equationt::SSA_stept &step, const smt_ast *ast)
{
  step.cond_ast = ast;
  return;
}

void
build_equation_class()
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
    .def_readwrite("assignment_type", &step::assignment_type)
    .def_readwrite("cond", &step::cond)
    .def_readwrite("comment", &step::comment)
    // For some reason, def_readwrite can't synthesize it's own setter
    // due to const perhaps, or smt_astt being opaque
    .add_property("guard_ast", make_getter(&step::guard_ast), make_function(&set_guard_ast))
    .add_property("cond_ast", make_getter(&step::cond_ast), make_function(&set_cond_ast))
    .def_readwrite("ignore", &step::ignore);

  class_<symex_targett, boost::shared_ptr<symex_targett>, boost::noncopyable>("symex_targett", no_init);

  class_<goto_symext::symex_resultt, boost::shared_ptr<goto_symext::symex_resultt> >("symex_resultt",
      init<boost::shared_ptr<symex_targett>, unsigned, unsigned>())
    .def_readwrite("target", &goto_symext::symex_resultt::target)
    .def_readwrite("total_claims", &goto_symext::symex_resultt::total_claims)
    .def_readwrite("remaining_claims", &goto_symext::symex_resultt::remaining_claims);

  init<const namespacet &> eq_init;
  class_<symex_target_equationt, boost::shared_ptr<symex_target_equationt>, bases<symex_targett> >("equation", eq_init)
    .def("assignment", &symex_target_equationt::assignment)
    .def("assumption", &symex_target_equationt::assumption)
    .def("assertion", &symex_target_equationt::assertion)
    .def("renumber", &symex_target_equationt::renumber)
    .def("convert", &symex_target_equationt::convert)
    .def("clear", &symex_target_equationt::clear)
    .def("check_for_dups", &symex_target_equationt::check_for_duplicate_assigns)
    .def("clone", &symex_target_equationt::clone)
    .def("clear_assertions", &symex_target_equationt::clear_assertions);
  return;
}
#endif
