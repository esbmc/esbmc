#include <cassert>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/witnesses.h>

expr2tc get_value(smt_convt &smt_conv, const expr2tc &expr)
{
  if(is_nil_expr(expr))
    return expr2tc();

  switch(expr->expr_id)
  {
    case expr2t::constant_int_id:
    case expr2t::constant_fixedbv_id:
    case expr2t::constant_floatbv_id:
    case expr2t::constant_bool_id:
    case expr2t::constant_string_id:
    case expr2t::constant_struct_id:
    case expr2t::constant_union_id:
    case expr2t::constant_array_id:
    case expr2t::constant_array_of_id:
      return expr;

    case expr2t::symbol_id:
      return smt_conv.get(expr);

    case expr2t::equality_id:
    {
      equality2t eq = to_equality2t(expr);

      expr2tc side1 = get_value(smt_conv, eq.side_1);
      if(is_nil_expr(side1)) break;

      expr2tc side2 = get_value(smt_conv, eq.side_2);
      if(is_nil_expr(side2)) break;

      equality2tc new_eq(side1, side2);
      simplify(new_eq);

      return new_eq;
    }

    case expr2t::not_id:
    {
      not2t n = to_not2t(expr);
      assert(is_bool_type(n.value));

      expr2tc value = get_value(smt_conv, n.value);
      if(is_nil_expr(value)) break;

      make_not(value);
      return value;
    }

    case expr2t::if_id:
    {
      if2t i = to_if2t(expr);

      expr2tc cond = get_value(smt_conv, i.cond);
      if(is_nil_expr(cond)) break;

      if(is_true(cond))
        return get_value(smt_conv, i.true_value);

      if(is_false(cond))
        return get_value(smt_conv, i.false_value);

      break;
    }

    case expr2t::with_id:
      return get_value(smt_conv, to_with2t(expr).update_value);

    case expr2t::typecast_id:
    {
      typecast2t t = to_typecast2t(expr);

      expr2tc from = get_value(smt_conv, t.from);
      if(is_nil_expr(from)) break;

      typecast2tc new_t(expr->type, from, t.rounding_mode);
      simplify(new_t);

      return new_t;
    }

    default:;
  }

  return expr2tc();
}

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

expr2tc build_value(smt_convt &smt_conv, const expr2tc &lhs, const expr2tc &rhs)
{
  if(is_nil_expr(rhs))
    return expr2tc();

  if(is_constant_expr(rhs))
    return rhs;

  if(is_symbol2t(rhs))
    return get_value(smt_conv, rhs);

  if(is_with2t(rhs))
    return get_value(smt_conv, to_with2t(rhs).update_value);

  (void) lhs;
  abort();
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
    goto_trace_step.value = build_value(smt_conv, SSA_step.lhs, SSA_step.rhs);

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
