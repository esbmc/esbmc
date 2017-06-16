#include <cassert>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/witnesses.h>

unsigned int get_member_name_field(const type2tc &t, const irep_idt &name)
{
  unsigned int idx = 0;
  const struct_union_data &data_ref =
    dynamic_cast<const struct_union_data &>(*t.get());

  for(auto const &it : data_ref.member_names)
  {
    if (it == name)
      break;
    idx++;
  }
  assert(idx != data_ref.member_names.size() &&
         "Member name of with expr not found in struct type");

  return idx;
}

expr2tc build_lhs(
  boost::shared_ptr<smt_convt> &smt_conv, const expr2tc &lhs)
{
  if(is_nil_expr(lhs))
    return lhs;

  expr2tc new_lhs = lhs;
  switch(new_lhs->expr_id)
  {
    case expr2t::index_id:
    {
      // An array subscription
      index2t index = to_index2t(lhs);

      // Build new source value, it might be an index, in case of
      // multidimensional arrays
      expr2tc new_source_value = build_lhs(smt_conv, index.source_value);
      expr2tc new_value = smt_conv->get(index.index);
      new_lhs = index2tc(lhs->type, new_source_value, new_value);
      break;
    }

    case expr2t::symbol_id:
    case expr2t::member_id:
      break;

    default:
      assert(0);
      break;
  }

  renaming::renaming_levelt::get_original_name(new_lhs, symbol2t::level0);
  return new_lhs;
}

expr2tc build_rhs(
  boost::shared_ptr<smt_convt> &smt_conv,
  const expr2tc &lhs,
  const expr2tc &rhs)
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
    {
      // An member access
      if(is_member2t(lhs))
      {
        member2t m = to_member2t(lhs);
        unsigned int v = get_member_name_field(rhs->type, m.member);
        new_rhs = is_constant_string2t(rhs) ?
          to_constant_struct2t(rhs).datatype_members[v] :
          to_constant_union2t(rhs).datatype_members[v];
      }

      // It should be an union/struct initialization
      break;
    }

    case expr2t::constant_array_of_id:
    case expr2t::if_id:
    case expr2t::symbol_id:
    case expr2t::bitcast_id:
    case expr2t::equality_id:
      break;

    default:
      rhs->dump();
      assert(0);
      break;
  }

  return smt_conv->get(new_rhs);
}

void build_goto_trace(
  const boost::shared_ptr<symex_target_equationt> target,
  boost::shared_ptr<smt_convt> &smt_conv,
  goto_tracet &goto_trace)
{
  unsigned step_nr = 0;

  for(auto SSA_step : target->SSA_steps)
  {
    tvt result = smt_conv->l_get(SSA_step.guard_ast);
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
    goto_trace_step.lhs = build_lhs(smt_conv, SSA_step.original_lhs);
    goto_trace_step.value = build_rhs(smt_conv, goto_trace_step.lhs, SSA_step.rhs);

    if(!is_nil_expr(SSA_step.lhs))
      goto_trace_step.value = smt_conv->get(SSA_step.lhs);

    for(auto it : SSA_step.converted_output_args)
    {
      if (is_constant_expr(it))
        goto_trace_step.output_args.push_back(it);
      else
        goto_trace_step.output_args.push_back(smt_conv->get(it));
    }

    if(SSA_step.is_assert() || SSA_step.is_assume())
      goto_trace_step.guard = !smt_conv->l_get(SSA_step.cond_ast).is_false();
  }
}

void build_successful_goto_trace(
    const boost::shared_ptr<symex_target_equationt> target,
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
