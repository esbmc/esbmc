/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>

#include <i2string.h>
#include <std_expr.h>
#include <expr_util.h>
#include <irep2.h>
#include <migrate.h>

#include <langapi/language_util.h>

#include "goto_symex_state.h"
#include "symex_target_equation.h"

void symex_target_equationt::assignment(
  const expr2tc &guard,
  const expr2tc &lhs,
  const expr2tc &original_lhs,
  const expr2tc &rhs,
  const sourcet &source,
  std::vector<dstring> stack_trace,
  assignment_typet assignment_type)
{
  assert(!is_nil_expr(lhs));

  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.lhs = lhs;
  SSA_step.original_lhs = original_lhs;
  SSA_step.rhs = rhs;
  SSA_step.assignment_type=assignment_type;
  SSA_step.cond = expr2tc(new equality2t(lhs, rhs));
  SSA_step.type=goto_trace_stept::ASSIGNMENT;
  SSA_step.source=source;
  SSA_step.stack_trace = stack_trace;
}

void symex_target_equationt::output(
  const expr2tc &guard,
  const sourcet &source,
  const std::string &fmt,
  const std::list<expr2tc> &args)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.type=goto_trace_stept::OUTPUT;
  SSA_step.source=source;
  SSA_step.output_args=args;
  SSA_step.format_string=fmt;
}

void symex_target_equationt::assumption(
  const expr2tc &guard,
  const expr2tc &cond,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type=goto_trace_stept::ASSUME;
  SSA_step.source=source;
}

void symex_target_equationt::assertion(
  const expr2tc &guard,
  const expr2tc &cond,
  const std::string &msg,
  std::vector<dstring> stack_trace,
  const sourcet &source)
{
  SSA_steps.push_back(SSA_stept());
  SSA_stept &SSA_step=SSA_steps.back();

  SSA_step.guard = guard;
  SSA_step.cond = cond;
  SSA_step.type=goto_trace_stept::ASSERT;
  SSA_step.source=source;
  SSA_step.comment=msg;
  SSA_step.stack_trace = stack_trace;
}


void symex_target_equationt::convert(prop_convt &prop_conv)
{
  bvt assertions;
  literalt assumpt_lit = const_literal(true);

  for (SSA_stepst::iterator it = SSA_steps.begin(); it != SSA_steps.end(); it++)
    convert_internal_step(prop_conv, assumpt_lit, assertions, *it);

  if (!assertions.empty())
    prop_conv.lcnf(assertions);

  return;
}

void symex_target_equationt::convert_internal_step(prop_convt &prop_conv,
                   literalt &assumpt_lit, bvt &assertions_lits, SSA_stept &step)
{
  static unsigned output_count = 0; // Temporary hack; should become scoped.
  bvt assert_bv;
  literalt true_lit = const_literal(true);
  literalt false_lit = const_literal(false);

  if (step.ignore) {
    step.cond_literal = true_lit;
    step.guard_literal = false_lit;
    return;
  }

  expr2tc tmp(step.guard);
  step.guard_literal = prop_conv.convert(tmp);

  if (step.is_assume() || step.is_assert()) {
    expr2tc tmp(step.cond);
    step.cond_literal = prop_conv.convert(tmp);
  } else if (step.is_assignment()) {
    expr2tc tmp2(step.cond);
    prop_conv.set_to(tmp2, true);
  } else if (step.is_output()) {
    for(std::list<expr2tc>::const_iterator
        o_it = step.output_args.begin();
        o_it != step.output_args.end();
        o_it++)
    {
      const expr2tc &tmp = *o_it;
      if(is_constant_expr(tmp) || is_constant_string2t(tmp))
        step.converted_output_args.push_back(tmp);
      else
      {
        expr2tc sym = expr2tc(new symbol2t(tmp->type,
                                 "symex::output::"+i2string(output_count++)));

        expr2tc eq = expr2tc(new equality2t(tmp, sym));
        prop_conv.set_to(eq, true);
        step.converted_output_args.push_back(sym);
      }
    }
  } else {
    assert(0 && "Unexpected SSA step type in conversion");
  }

  if (step.is_assert()) {
    step.cond_literal = prop_conv.limplies(assumpt_lit, step.cond_literal);
    assertions_lits.push_back(prop_conv.lnot(step.cond_literal));
  } else if (step.is_assume()) {
    assumpt_lit = prop_conv.land(assumpt_lit, step.cond_literal);
  }

  return;
}

void symex_target_equationt::output(std::ostream &out) const
{
  for(SSA_stepst::const_iterator
      it=SSA_steps.begin();
      it!=SSA_steps.end();
      it++)
  {
    it->output(ns, out);
    out << "--------------" << std::endl;
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
  case goto_trace_stept::ASSERT: out << "ASSERT" << std::endl; break;
  case goto_trace_stept::ASSUME: out << "ASSUME" << std::endl; break;
  case goto_trace_stept::OUTPUT: out << "OUTPUT" << std::endl; break;

  case goto_trace_stept::ASSIGNMENT:
    out << "ASSIGNMENT (";
    switch(assignment_type)
    {
    case HIDDEN: out << "HIDDEN"; break;
    case STATE: out << "STATE"; break;
    default:;
    }

    out << ")" << std::endl;
    break;

  default: assert(false);
  }

  if(is_assert() || is_assume() || is_assignment())
    out << from_expr(ns, "", migrate_expr_back(cond)) << std::endl;

  if(is_assert())
    out << comment << std::endl;

  out << "Guard: " << from_expr(ns, "", migrate_expr_back(guard)) << std::endl;
}

void
symex_target_equationt::push_ctx(void)
{
}

void
symex_target_equationt::pop_ctx(void)
{
}

std::ostream &operator<<(
  std::ostream &out,
  const symex_target_equationt &equation)
{
  equation.output(out);
  return out;
}

runtime_encoded_equationt::runtime_encoded_equationt(const namespacet &_ns,
                                                     prop_convt &_conv)
  : symex_target_equationt(_ns),
    conv(_conv)
{
  assert_vec_list.push_back(bvt());
  assumpt_chain.push_back(const_literal(true));
}

void
runtime_encoded_equationt::flush_latest_instructions(void)
{
  SSA_stepst::iterator run_it = scoped_end_points.back();

  // Convert this run.
  if (SSA_steps.size() != 0) {
    // Horror: if the start-of-run iterator is end, then it actually refers to
    // the start of the list. The start doesn't have a persistent iterator, so
    // we can't keep a reference to it when there's nothing in the list :|
    if (run_it == SSA_steps.end())
      run_it = SSA_steps.begin();
    for (; run_it != SSA_steps.end(); run_it++)
      convert_internal_step(conv, assumpt_chain.back(), assert_vec_list.back(),
                            *run_it);
  }
}

void
runtime_encoded_equationt::push_ctx(void)
{

  flush_latest_instructions();

  SSA_stepst::iterator it = SSA_steps.end();

  if (SSA_steps.size() != 0)
    --it;

  // And push everything back.
  assumpt_chain.push_back(assumpt_chain.back());
  assert_vec_list.push_back(assert_vec_list.back());
  scoped_end_points.push_back(it);
  conv.push_ctx();
}

void
runtime_encoded_equationt::pop_ctx(void)
{

  SSA_stepst::iterator it = scoped_end_points.back();

  if (SSA_steps.size() != 0)
    ++it;

  SSA_steps.erase(it, SSA_steps.end());

  conv.pop_ctx();
  scoped_end_points.pop_back();
  assert_vec_list.pop_back();
  assumpt_chain.pop_back();
}

void
runtime_encoded_equationt::convert(prop_convt &prop_conv)
{

  // Don't actually convert. We've already done most of the conversion by now
  // (probably), instead flush all unconverted instructions. We don't push
  // a context, because a) where do we unpop it, but b) we're never going to
  // build anything on top of this, so there's no gain by pushing it.
  flush_latest_instructions();

  // Finally, we also want to assert the set of assertions.
  if(!assert_vec_list.back().empty())
    prop_conv.lcnf(assert_vec_list.back());

  return;
}

symex_targett *
runtime_encoded_equationt::clone(void) const
{
  assert(0 && "runtime_encoded_equationt should never be cloned");
}
