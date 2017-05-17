#include <cassert>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/witnesses.h>

expr2tc build_lhs(smt_convt &smt_conv, const expr2tc &lhs, const expr2tc &rhs)
{
  if(is_nil_expr(lhs))
    return expr2tc();

  if(is_symbol2t(lhs))
    return lhs;

  if(is_index2t(lhs))
  {
    // The rhs must be a with statement
    assert(is_with2t(rhs));

    // Get value
    with2t new_rhs = to_with2t(rhs);
    expr2tc value = get_value(smt_conv, new_rhs.update_field);
    assert(!is_nil_expr(value));

    // Construct new index
    return index2tc(lhs->type, to_index2t(lhs).source_value, value);
  }

  return expr2tc();
}

void build_goto_trace(
  const symex_target_equationt &target,
  smt_convt &smt_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;

  for(auto SSA_step : target.SSA_steps)
  {
    tvt result = smt_conv.l_get(SSA_step.guard_ast);
    if(!result.is_true())
      continue;

    if(SSA_step.assignment_type == symex_target_equationt::HIDDEN
       && SSA_step.is_assignment())
      continue;

    step_nr++;

    goto_trace.steps.push_back(goto_trace_stept());
    goto_trace_stept &goto_trace_step = goto_trace.steps.back();

    goto_trace_step.thread_nr = SSA_step.source.thread_nr;
    goto_trace_step.pc = SSA_step.source.pc;
    goto_trace_step.comment = SSA_step.comment;
    goto_trace_step.original_lhs = SSA_step.original_lhs;
    goto_trace_step.type = SSA_step.type;
    goto_trace_step.step_nr = step_nr;
    goto_trace_step.format_string = SSA_step.format_string;
    goto_trace_step.stack_trace = SSA_step.stack_trace;
    goto_trace_step.lhs =
      build_lhs(smt_conv, SSA_step.original_lhs, SSA_step.rhs);

    goto_trace_step.value = get_value(smt_conv, SSA_step.rhs);

    for(auto it : SSA_step.converted_output_args)
    {
      if (is_constant_expr(it))
        goto_trace_step.output_args.push_back(it);
      else
        goto_trace_step.output_args.push_back(smt_conv.get(it));
    }

    if(SSA_step.is_assert() || SSA_step.is_assume())
      goto_trace_step.guard = !smt_conv.l_get(SSA_step.cond_ast).is_false();
  }
}

void build_successful_goto_trace(
    const symex_target_equationt &target,
    const namespacet &ns,
    goto_tracet &goto_trace)
{
  unsigned step_nr=0;
  for(symex_target_equationt::SSA_stepst::const_iterator
      it=target.SSA_steps.begin();
      it!=target.SSA_steps.end(); it++)
  {
    if((it->is_assignment() || it->is_assert() || it->is_assume())
      && (is_valid_witness_expr(ns, it->lhs)))
    {
      goto_trace.steps.push_back(goto_trace_stept());
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
