#include <cassert>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/witnesses.h>

expr2tc build_lhs(boost::shared_ptr<smt_convt> &smt_conv, const expr2tc &lhs)
{
  if(is_nil_expr(lhs))
    return lhs;

  expr2tc new_lhs = lhs;
  switch(new_lhs->expr_id)
  {
  case expr2t::index_id:
  {
    // An array subscription
    index2t index = to_index2t(new_lhs);

    // Build new source value, it might be an index, in case of
    // multidimensional arrays
    expr2tc new_source_value = build_lhs(smt_conv, index.source_value);
    expr2tc new_value = smt_conv->get(index.index);
    new_lhs = index2tc(new_lhs->type, new_source_value, new_value);
    break;
  }

  case expr2t::typecast_id:
    new_lhs = to_typecast2t(new_lhs).from;
    break;

  case expr2t::bitcast_id:
    new_lhs = to_bitcast2t(new_lhs).from;
    break;

  default:
    break;
  }

  renaming::renaming_levelt::get_original_name(new_lhs, symbol2t::level0);
  return new_lhs;
}

expr2tc build_rhs(boost::shared_ptr<smt_convt> &smt_conv, const expr2tc &rhs)
{
  if(is_nil_expr(rhs) || is_constant_expr(rhs))
    return rhs;

  auto new_rhs = smt_conv->get(rhs);
  renaming::renaming_levelt::get_original_name(new_rhs, symbol2t::level0);
  return new_rhs;
}

void build_goto_trace(
  const boost::shared_ptr<symex_target_equationt> &target,
  boost::shared_ptr<smt_convt> &smt_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;

  for(auto SSA_step : target->SSA_steps)
  {
    tvt result = smt_conv->l_get(SSA_step.guard_ast);
    if(!result.is_true())
      continue;

    if(SSA_step.hidden)
      continue;

    goto_trace.steps.emplace_back();
    goto_trace_stept &goto_trace_step = goto_trace.steps.back();

    goto_trace_step.thread_nr = SSA_step.source.thread_nr;
    goto_trace_step.pc = SSA_step.source.pc;
    goto_trace_step.comment = SSA_step.comment;
    goto_trace_step.original_lhs = SSA_step.original_lhs;
    goto_trace_step.type = SSA_step.type;
    goto_trace_step.step_nr = ++step_nr;
    goto_trace_step.format_string = SSA_step.format_string;

    goto_trace_step.stack_trace = SSA_step.stack_trace;

    if(SSA_step.is_assignment())
    {
      goto_trace_step.lhs = build_lhs(smt_conv, SSA_step.original_lhs);
      goto_trace_step.value = build_rhs(smt_conv, SSA_step.rhs);
    }

    if(SSA_step.is_output())
    {
      for(const auto &arg : SSA_step.converted_output_args)
      {
        if(is_constant_expr(arg))
          goto_trace_step.output_args.push_back(arg);
        else
          goto_trace_step.output_args.push_back(smt_conv->get(arg));
      }
    }

    if(SSA_step.is_assert() || SSA_step.is_assume())
      goto_trace_step.guard = !smt_conv->l_get(SSA_step.cond_ast).is_false();
  }
}

void build_successful_goto_trace(
  const boost::shared_ptr<symex_target_equationt> &target,
  const namespacet &ns,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;
  for(symex_target_equationt::SSA_stepst::const_iterator it =
        target->SSA_steps.begin();
      it != target->SSA_steps.end();
      it++)
  {
    if(
      (it->is_assert() || it->is_assume()) &&
      (is_valid_witness_expr(ns, it->lhs)))
    {
      // When building the correctness witness, we only care about
      // asserts and assumes
      if(!(it->is_assert() || it->is_assume()))
        continue;

      goto_trace.steps.emplace_back();
      goto_trace_stept &goto_trace_step = goto_trace.steps.back();
      goto_trace_step.thread_nr = it->source.thread_nr;
      goto_trace_step.lhs = it->lhs;
      goto_trace_step.rhs = it->rhs;
      goto_trace_step.pc = it->source.pc;
      goto_trace_step.comment = it->comment;
      goto_trace_step.original_lhs = it->original_lhs;
      goto_trace_step.type = it->type;
      goto_trace_step.step_nr = step_nr++;
      goto_trace_step.format_string = it->format_string;
      goto_trace_step.stack_trace = it->stack_trace;
    }
  }
}
