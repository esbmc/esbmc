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

expr2tc build_rhs(smt_convt &smt_conv, const expr2tc &lhs, const expr2tc &rhs)
{
  if(is_nil_expr(rhs))
    return rhs;

  expr2tc new_rhs = rhs;
  switch(rhs->expr_id)
  {
    case expr2t::constant_int_id:
    case expr2t::constant_fixedbv_id:
    case expr2t::constant_floatbv_id:
    case expr2t::constant_bool_id:
    case expr2t::constant_string_id:
      return rhs;

    case expr2t::constant_array_id:
    {
      // An array subscription, we should be able to get the value directly,
      // as lhs should have been resolved already
      if(is_index2t(lhs))
      {
        index2t i = to_index2t(lhs);
        assert(is_bv_type(i.index));

        constant_int2t v = to_constant_int2t(i.index);
        new_rhs = to_constant_array2t(rhs).datatype_members[v.value.to_uint64()];
      }

      // It should be an array initialization
      break;
    }

    case expr2t::with_id:
      new_rhs = to_with2t(rhs).update_value;
      break;

    case expr2t::constant_struct_id:
    case expr2t::constant_union_id:
    case expr2t::constant_array_of_id:
    case expr2t::if_id:
    case expr2t::symbol_id:
      break;

    default:
      abort();
  }

  return smt_conv.get(new_rhs);
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
    goto_trace_step.value = build_rhs(smt_conv, goto_trace_step.lhs, SSA_step.rhs);

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
