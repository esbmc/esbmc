/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_symex_state.h>
#include <goto-symex/symex_target_equation.h>
#include <langapi/language_util.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/std_expr.h>

void symex_target_equationt::assignment(
  const expr2tc &guard,
  const expr2tc &lhs,
  const expr2tc &original_lhs,
  const expr2tc &rhs,
  const sourcet &source,
  std::vector<stack_framet> stack_trace,
  assignment_typet assignment_type)
{
  assert(!is_nil_expr(lhs));

  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = lhs;
  SSA_step.original_lhs = original_lhs;
  SSA_step.rhs = rhs;
  SSA_step.assignment_type = assignment_type;
  SSA_step.cond = equality2tc(lhs, rhs);
  SSA_step.type = goto_trace_stept::ASSIGNMENT;
  SSA_step.source = source;
  SSA_step.stack_trace = stack_trace;

  if(debug_print)
    SSA_step.short_output(ns, std::cout);
}

void symex_target_equationt::output(
  const expr2tc &guard,
  const sourcet &source,
  const std::string &fmt,
  const std::list<expr2tc> &args)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.type = goto_trace_stept::OUTPUT;
  SSA_step.source = source;
  SSA_step.output_args = args;
  SSA_step.format_string = fmt;

  if(debug_print)
    SSA_step.short_output(ns, std::cout);
}

void symex_target_equationt::assumption(
  const expr2tc &guard,
  const expr2tc &cond,
  const sourcet &source)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type = goto_trace_stept::ASSUME;
  SSA_step.source = source;

  if(debug_print)
    SSA_step.short_output(ns, std::cout);
}

void symex_target_equationt::assertion(
  const expr2tc &guard,
  const expr2tc &cond,
  const std::string &msg,
  std::vector<stack_framet> stack_trace,
  const sourcet &source)
{
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type = goto_trace_stept::ASSERT;
  SSA_step.source = source;
  SSA_step.comment = msg;
  SSA_step.stack_trace = stack_trace;

  if(debug_print)
    SSA_step.short_output(ns, std::cout);
}

void symex_target_equationt::renumber(
  const expr2tc &guard,
  const expr2tc &symbol,
  const expr2tc &size,
  const sourcet &source)
{
  assert(is_symbol2t(symbol));
  assert(is_bv_type(size));
  SSA_steps.emplace_back();
  SSA_stept &SSA_step = SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = symbol;
  SSA_step.rhs = size;
  SSA_step.type = goto_trace_stept::RENUMBER;
  SSA_step.source = source;

  if(debug_print)
    SSA_step.short_output(ns, std::cout);
}

void symex_target_equationt::convert(smt_convt &smt_conv)
{
  smt_convt::ast_vec assertions;
  const smt_ast *assumpt_ast = smt_conv.convert_ast(gen_true_expr());

  for(auto &SSA_step : SSA_steps)
    convert_internal_step(smt_conv, assumpt_ast, assertions, SSA_step);

  if(!assertions.empty())
    smt_conv.assert_ast(
      smt_conv.make_n_ary(&smt_conv, &smt_convt::mk_or, assertions));
}

void symex_target_equationt::convert_internal_step(
  smt_convt &smt_conv,
  const smt_ast *&assumpt_ast,
  smt_convt::ast_vec &assertions,
  SSA_stept &step)
{
  static unsigned output_count = 0; // Temporary hack; should become scoped.
  bvt assert_bv;
  const smt_ast *true_val = smt_conv.convert_ast(gen_true_expr());
  const smt_ast *false_val = smt_conv.convert_ast(gen_false_expr());

  if(step.ignore)
  {
    step.cond_ast = true_val;
    step.guard_ast = false_val;
    return;
  }

  if(ssa_trace)
  {
    step.output(ns, std::cout);
    std::cout << std::endl;
  }

  expr2tc tmp(step.guard);
  step.guard_ast = smt_conv.convert_ast(tmp);

  if(step.is_assume() || step.is_assert())
  {
    expr2tc tmp(step.cond);
    step.cond_ast = smt_conv.convert_ast(tmp);

    if(ssa_smt_trace)
    {
      step.cond_ast->dump();
      std::cout << std::endl;
    }
  }
  else if(step.is_assignment())
  {
    smt_astt assign = smt_conv.convert_assign(step.cond);
    if(ssa_smt_trace)
    {
      assign->dump();
      std::cout << std::endl;
    }
  }
  else if(step.is_output())
  {
    for(std::list<expr2tc>::const_iterator o_it = step.output_args.begin();
        o_it != step.output_args.end();
        o_it++)
    {
      const expr2tc &tmp = *o_it;
      if(is_constant_expr(tmp) || is_constant_string2t(tmp))
        step.converted_output_args.push_back(tmp);
      else
      {
        symbol2tc sym(tmp->type, "symex::output::" + i2string(output_count++));
        equality2tc eq(sym, tmp);
        smt_conv.set_to(eq, true);
        step.converted_output_args.push_back(sym);
      }
    }
  }
  else if(step.is_renumber())
  {
    smt_conv.renumber_symbol_address(step.guard, step.lhs, step.rhs);
  }
  else if(!step.is_skip())
  {
    assert(0 && "Unexpected SSA step type in conversion");
  }

  if(step.is_assert())
  {
    step.cond_ast = smt_conv.imply_ast(assumpt_ast, step.cond_ast);
    assertions.push_back(smt_conv.invert_ast(step.cond_ast));
  }
  else if(step.is_assume())
  {
    smt_convt::ast_vec v;
    v.push_back(assumpt_ast);
    v.push_back(step.cond_ast);
    assumpt_ast = smt_conv.make_n_ary(&smt_conv, &smt_convt::mk_and, v);
  }
}

void symex_target_equationt::output(std::ostream &out) const
{
  for(const auto &SSA_step : SSA_steps)
  {
    SSA_step.output(ns, out);
    out << "--------------" << std::endl;
  }
}

void symex_target_equationt::short_output(std::ostream &out, bool show_ignored)
  const
{
  for(const auto &SSA_step : SSA_steps)
  {
    SSA_step.short_output(ns, out, show_ignored);
  }
}

void symex_target_equationt::SSA_stept::output(
  const namespacet &ns,
  std::ostream &out) const
{
  if(source.is_set)
  {
    out << "Thread " << source.thread_nr;

    if(source.pc->location.is_not_nil())
      out << " " << source.pc->location << std::endl;
    else
      out << std::endl;
  }

  switch(type)
  {
  case goto_trace_stept::ASSERT:
    out << "ASSERT" << std::endl;
    break;
  case goto_trace_stept::ASSUME:
    out << "ASSUME" << std::endl;
    break;
  case goto_trace_stept::OUTPUT:
    out << "OUTPUT" << std::endl;
    break;

  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT (";
    switch(assignment_type)
    {
    case HIDDEN:
      out << "HIDDEN";
      break;
    case STATE:
      out << "STATE";
      break;
    default:;
    }

    out << ")" << std::endl;
    break;

  default:
    assert(false);
  }

  if(is_assert() || is_assume() || is_assignment())
    out << from_expr(ns, "", migrate_expr_back(cond)) << std::endl;

  if(is_assert())
    out << comment << std::endl;

  if(config.options.get_bool_option("show-guards"))
    out << "Guard: " << from_expr(ns, "", migrate_expr_back(guard))
        << std::endl;
}

void symex_target_equationt::SSA_stept::short_output(
  const namespacet &ns,
  std::ostream &out,
  bool show_ignored) const
{
  if((is_assignment() || is_assert() || is_assume()) && show_ignored == ignore)
  {
    out << from_expr(ns, "", cond) << std::endl;
  }
  else if(is_renumber())
  {
    out << "renumber: " << from_expr(ns, "", lhs) << std::endl;
  }
}

void symex_target_equationt::push_ctx()
{
}

void symex_target_equationt::pop_ctx()
{
}

std::ostream &
operator<<(std::ostream &out, const symex_target_equationt &equation)
{
  equation.output(out);
  return out;
}

void symex_target_equationt::check_for_duplicate_assigns() const
{
  std::map<std::string, unsigned int> countmap;
  unsigned int i = 0;

  for(const auto &SSA_step : SSA_steps)
  {
    i++;
    if(!SSA_step.is_assignment())
      continue;

    const equality2t &ref = to_equality2t(SSA_step.cond);
    const symbol2t &sym = to_symbol2t(ref.side_1);
    countmap[sym.get_symbol_name()]++;
  }

  for(std::map<std::string, unsigned int>::const_iterator it = countmap.begin();
      it != countmap.end();
      it++)
  {
    if(it->second != 1)
    {
      std::cerr << "Symbol \"" << it->first << "\" appears " << it->second
                << " times" << std::endl;
    }
  }

  std::cerr << "Checked " << i << " insns" << std::endl;
}

unsigned int symex_target_equationt::clear_assertions()
{
  unsigned int num_asserts = 0;

  for(SSA_stepst::iterator it = SSA_steps.begin(); it != SSA_steps.end(); it++)
  {
    if(it->type == goto_trace_stept::ASSERT)
    {
      SSA_stepst::iterator it2 = it;
      it--;
      SSA_steps.erase(it2);
      num_asserts++;
    }
  }

  return num_asserts;
}

runtime_encoded_equationt::runtime_encoded_equationt(
  const namespacet &_ns,
  smt_convt &_conv)
  : symex_target_equationt(_ns), conv(_conv)
{
  assert_vec_list.emplace_back();
  assumpt_chain.push_back(conv.convert_ast(gen_true_expr()));
  cvt_progress = SSA_steps.end();
}

void runtime_encoded_equationt::flush_latest_instructions()
{
  if(SSA_steps.size() == 0)
    return;

  SSA_stepst::iterator run_it = cvt_progress;
  // Scenarios:
  // * We're at the start of running, in which case cvt_progress == end
  // * We're in the middle, but nothing is left to push, so run_it + 1 == end
  // * We're in the middle, and there's more to convert.
  if(run_it == SSA_steps.end())
  {
    run_it = SSA_steps.begin();
  }
  else
  {
    run_it++;
    if(run_it == SSA_steps.end())
    {
      // There is in fact, nothing to do
      return;
    }

    // Just roll on
  }

  // Now iterate from the start insn to convert, to the end of the list.
  for(; run_it != SSA_steps.end(); ++run_it)
    convert_internal_step(
      conv, assumpt_chain.back(), assert_vec_list.back(), *run_it);

  run_it--;
  cvt_progress = run_it;
}

void runtime_encoded_equationt::push_ctx()
{
  flush_latest_instructions();

  // And push everything back.
  assumpt_chain.push_back(assumpt_chain.back());
  assert_vec_list.push_back(assert_vec_list.back());
  scoped_end_points.push_back(cvt_progress);
  conv.push_ctx();
}

void runtime_encoded_equationt::pop_ctx()
{
  SSA_stepst::iterator it = scoped_end_points.back();
  cvt_progress = it;

  if(SSA_steps.size() != 0)
    ++it;

  SSA_steps.erase(it, SSA_steps.end());

  conv.pop_ctx();
  scoped_end_points.pop_back();
  assert_vec_list.pop_back();
  assumpt_chain.pop_back();
}

void runtime_encoded_equationt::convert(smt_convt &smt_conv)
{
  // Don't actually convert. We've already done most of the conversion by now
  // (probably), instead flush all unconverted instructions. We don't push
  // a context, because a) where do we unpop it, but b) we're never going to
  // build anything on top of this, so there's no gain by pushing it.
  flush_latest_instructions();

  // Finally, we also want to assert the set of assertions.
  if(!assert_vec_list.back().empty())
    smt_conv.assert_ast(smt_conv.make_n_ary(
      &smt_conv, &smt_convt::mk_or, assert_vec_list.back()));
}

boost::shared_ptr<symex_targett> runtime_encoded_equationt::clone() const
{
  // Only permit cloning at the start of a run - there should never be any data
  // in this formula when it happens. Cloning needs to be supported so that a
  // reachability_treet can take a template equation and clone it ever time it
  // sets up a new exploration.
  assert(
    SSA_steps.size() == 0 &&
    "runtime_encoded_equationt shouldn't be "
    "cloned when it contains data");
  auto nthis = boost::shared_ptr<runtime_encoded_equationt>(
    new runtime_encoded_equationt(*this));
  nthis->cvt_progress = nthis->SSA_steps.end();
  return nthis;
}

tvt runtime_encoded_equationt::ask_solver_question(const expr2tc &question)
{
  tvt final_res;

  // So - we have a formula, we want to work out whether it's true, false, or
  // unknown. Before doing anything, first push a context, as we'll need to
  // wipe some state afterwards.
  push_ctx();

  // Convert the question (must be a bool).
  assert(is_bool_type(question));
  const smt_ast *q = conv.convert_ast(question);

  // The proposition also needs to be guarded with the in-program assumptions,
  // which are not necessarily going to be part of the state guard.
  conv.assert_ast(assumpt_chain.back());

  // Now, how to ask the question? Unfortunately the clever solver stuff won't
  // negate the condition, it'll only give us a handle to it that it negates
  // when we access. So, we have to make an assertion, check it, pop it, then
  // check another.
  // Those assertions are just is-the-prop-true, is-the-prop-false. Valid
  // results are true, false, both.
  push_ctx();
  conv.assert_ast(q);
  smt_convt::resultt res1 = conv.dec_solve();
  pop_ctx();
  push_ctx();
  conv.assert_ast(conv.invert_ast(q));
  smt_convt::resultt res2 = conv.dec_solve();
  pop_ctx();

  // So; which result?
  if(
    res1 == smt_convt::P_ERROR || res1 == smt_convt::P_SMTLIB ||
    res2 == smt_convt::P_ERROR || res2 == smt_convt::P_SMTLIB)
  {
    std::cerr << "Solver returned error while asking question" << std::endl;
    abort();
  }
  else if(res1 == smt_convt::P_SATISFIABLE && res2 == smt_convt::P_SATISFIABLE)
  {
    // Both ways are satisfiable; result is unknown.
    final_res = tvt(tvt::TV_UNKNOWN);
  }
  else if(
    res1 == smt_convt::P_SATISFIABLE && res2 == smt_convt::P_UNSATISFIABLE)
  {
    // Truth of question is satisfiable; other not; so we're true.
    final_res = tvt(tvt::TV_TRUE);
  }
  else if(
    res1 == smt_convt::P_UNSATISFIABLE && res2 == smt_convt::P_SATISFIABLE)
  {
    // Truth is unsat, false is sat, proposition is false
    final_res = tvt(tvt::TV_FALSE);
  }
  else
  {
    pop_ctx();
    throw dual_unsat_exception();
  }

  // We have our result; pop off the questions / formula we've asked.
  pop_ctx();

  return final_res;
}
