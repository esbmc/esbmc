/*******************************************************************\

Module: Traces of GOTO Programs

Author: Daniel Kroening

  Date: July 2005

\*******************************************************************/

#include <cassert>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/witnesses.h>

void build_goto_trace(
  const boost::shared_ptr<symex_target_equationt>& target,
  boost::shared_ptr<smt_convt> &smt_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr=0;

  for(symex_target_equationt::SSA_stepst::const_iterator
      it=target->SSA_steps.begin();
      it!=target->SSA_steps.end();
      it++)
  {

    const symex_target_equationt::SSA_stept &SSA_step=*it;
    tvt result = smt_conv->l_get(SSA_step.guard_ast);

    if(result != tvt(true))
      continue;

    if(SSA_step.assignment_type == symex_target_equationt::HIDDEN
       && it->is_assignment())
      continue;

    step_nr++;

    goto_trace.steps.emplace_back();
    goto_trace_stept &goto_trace_step=goto_trace.steps.back();

    goto_trace_step.thread_nr=SSA_step.source.thread_nr;
    goto_trace_step.lhs=SSA_step.lhs;
    goto_trace_step.rhs=SSA_step.rhs;
    goto_trace_step.pc=SSA_step.source.pc;
    goto_trace_step.comment=SSA_step.comment;
    goto_trace_step.original_lhs=SSA_step.original_lhs;
    goto_trace_step.type=SSA_step.type;
    goto_trace_step.step_nr=step_nr;
    goto_trace_step.format_string=SSA_step.format_string;
    goto_trace_step.stack_trace = SSA_step.stack_trace;

    if(!is_nil_expr(SSA_step.lhs))
      goto_trace_step.value = smt_conv->get(SSA_step.lhs);

    for(const auto & arg : SSA_step.converted_output_args)
    {
      if (is_constant_expr(arg))
        goto_trace_step.output_args.push_back(arg);
      else
        goto_trace_step.output_args.push_back(smt_conv->get(arg));
    }

    if(SSA_step.is_assert() || SSA_step.is_assume())
      goto_trace_step.guard = !smt_conv->l_get(SSA_step.cond_ast).is_false();
  }
}

void build_successful_goto_trace(
    const boost::shared_ptr<symex_target_equationt>& target,
    const namespacet &ns,
    goto_tracet &goto_trace)
{
  unsigned step_nr=0;
  for(symex_target_equationt::SSA_stepst::const_iterator
      it=target->SSA_steps.begin();
      it!=target->SSA_steps.end(); it++)
  {
    if((it->is_assignment() || it->is_assert() || it->is_assume())
      && (is_valid_witness_expr(ns, it->lhs)))
    {
      goto_trace.steps.emplace_back();
      goto_trace_stept &goto_trace_step=goto_trace.steps.back();
      goto_trace_step.thread_nr=it->source.thread_nr;
      goto_trace_step.lhs=it->lhs;
      goto_trace_step.rhs=it->rhs;
      goto_trace_step.pc=it->source.pc;
      goto_trace_step.comment=it->comment;
      goto_trace_step.original_lhs=it->original_lhs;
      goto_trace_step.type=it->type;
      goto_trace_step.step_nr=step_nr;
      goto_trace_step.format_string=it->format_string;
      goto_trace_step.stack_trace = it->stack_trace;
    }
  }
}
