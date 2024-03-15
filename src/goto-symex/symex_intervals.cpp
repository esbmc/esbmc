#include <util/threeval.h>
#include <goto-symex/goto_symex.h>

bool symex_contains_unsupported(const expr2tc &e)
{
  if (
    is_floatbv_type(e) || is_fixedbv_type(e) || is_pointer_type(e) ||
    is_structure_type(e) || is_vector_type(e) || is_array_type(e))
    return true;

  bool result = false;
  e->foreach_operand([&result](const expr2tc &o) {
    if (!result)
      result |= symex_contains_unsupported(o);
  });
  return result;
}

tvt goto_symext::eval_boolean_expression(const expr2tc &cond) const
{
  // TODO: cache :)

  // TODO: Deal with floats :(
  if (symex_contains_unsupported(cond))
    return tvt(tvt::TV_UNKNOWN);

  wrapped_interval interval = get_interval(cond);
  // If the interval does not contain zero then it's always true
  if (!interval.contains(0))
    return tvt(tvt::TV_TRUE);

  // If it does contain zero and its singleton then it's always false
  if (interval.singleton())
    return tvt(tvt::TV_FALSE);

  return tvt(tvt::TV_UNKNOWN);
}

wrapped_interval
goto_symext::get_interval_from_symbol(const symbol2t &sym) const
{
  auto it = intervals.find(sym.get_symbol_name());
  return it != intervals.end() ? it->second : wrapped_interval(sym.type);
}
wrapped_interval goto_symext::get_interval(const expr2tc &e) const
{
  wrapped_interval result(e->type);

  switch (e->expr_id)
  {
  case expr2t::constant_bool_id:
    result.set_lower(to_constant_bool2t(e).is_true());
    result.set_upper(to_constant_bool2t(e).is_true());
    break;

  case expr2t::constant_int_id:
  {
    auto value = to_constant_int2t(e).value;
    result.set_lower(value);
    result.set_upper(value);
    assert(!result.is_bottom());
    break;
  }

  case expr2t::symbol_id:
    result = get_interval_from_symbol(to_symbol2t(e));
    break;

  case expr2t::neg_id:
    result = -get_interval(to_neg2t(e).value);
    break;

  case expr2t::not_id:
    result = wrapped_interval::invert_bool(get_interval(to_not2t(e).value));
    break;

  case expr2t::or_id:
  case expr2t::and_id:
  case expr2t::xor_id:
  case expr2t::implies_id:
  {
    const auto &logic_op = dynamic_cast<const logic_2ops &>(*e);
    tvt lhs = eval_boolean_expression(logic_op.side_1);
    tvt rhs = eval_boolean_expression(logic_op.side_2);

    if (is_and2t(e))
    {
      // If any side is false, (and => false)
      if ((lhs.is_false() || rhs.is_false()))
      {
        result.set_lower(0);
        result.set_upper(0);
        break;
      }

      // Both sides are true, then true
      if (lhs.is_true() && rhs.is_true())
      {
        result.set_lower(1);
        result.set_upper(1);
        break;
      }
    }

    else if (is_or2t(e))
    {
      if (lhs.is_true() || rhs.is_true())
      {
        result.set_lower(1);
        result.set_upper(1);
        break;
      }

      // Both sides are false, then false
      if (lhs.is_false() && rhs.is_false())
      {
        result.set_lower(0);
        result.set_upper(0);
        break;
      }
    }

    else if (is_xor2t(e))
    {
      if (lhs.is_unknown() || rhs.is_unknown())
      {
        result.set_lower(0);
        result.set_upper(1);
      }
      else if (lhs == rhs)
      {
        result.set_lower(0);
        result.set_upper(0);
      }
      else
      {
        result.set_lower(1);
        result.set_upper(1);
      }

      break;
    }

    else if (is_implies2t(e))
    {
      // A --> B <=== > ¬A or B
      if (lhs.is_true() && rhs.is_false())
      {
        result.set_lower(0);
        result.set_upper(0);
      }
      else if (lhs.is_false() || rhs.is_true())
      {
        result.set_lower(1);
        result.set_upper(1);
      }
      else
      {
        result.set_lower(0);
        result.set_upper(1);
      }
      break;
    }

    log_debug("interval", "Could not simplify: {}", *e);
    break;
  }

  case expr2t::if_id:
  {
    auto cond = get_interval(to_if2t(e).cond);
    auto lhs = get_interval(to_if2t(e).true_value);
    auto rhs = get_interval(to_if2t(e).false_value);
    result = wrapped_interval::ternary_if(cond, lhs, rhs);
    break;
  }
  case expr2t::typecast_id:
  {
    // Special case: boolean
    if (is_bool_type(to_typecast2t(e).type))
    {
      tvt truth = eval_boolean_expression(to_typecast2t(e).from);
      result.set_lower(0);
      result.set_upper(1);

      if (truth.is_true())
        result.set_lower(1);

      if (truth.is_false())
        result.set_upper(0);

      break;
    }
    auto inner = get_interval(to_typecast2t(e).from);
    result = wrapped_interval::cast(inner, to_typecast2t(e).type);
    break;
  }
  case expr2t::add_id:
  case expr2t::sub_id:
  case expr2t::mul_id:
  case expr2t::div_id:
  case expr2t::modulus_id:
  {
    const auto &arith_op = dynamic_cast<const arith_2ops &>(*e);
    e->dump();
    auto lhs = get_interval(arith_op.side_1);
    auto rhs = get_interval(arith_op.side_2);

    if (is_add2t(e))
      result = lhs + rhs;

    else if (is_sub2t(e))
      result = lhs - rhs;

    else if (is_mul2t(e))
      result = lhs * rhs;

    else if (is_div2t(e))
      result = lhs / rhs;

    else if (is_modulus2t(e))
      result = lhs % rhs;
  }
  break;

  case expr2t::shl_id:
  case expr2t::ashr_id:
  case expr2t::lshr_id:
  case expr2t::bitor_id:
  case expr2t::bitand_id:
  case expr2t::bitxor_id:
  case expr2t::bitnand_id:
  case expr2t::bitnor_id:
  case expr2t::bitnxor_id:
  {
    const auto &bit_op = dynamic_cast<const bit_2ops &>(*e);
    auto lhs = get_interval(bit_op.side_1);
    auto rhs = get_interval(bit_op.side_2);

    if (is_shl2t(e))
      result = wrapped_interval::left_shift(lhs, rhs);
    else if (is_ashr2t(e))
      result = wrapped_interval::arithmetic_right_shift(lhs, rhs);

    else if (is_lshr2t(e))
      result = wrapped_interval::logical_right_shift(lhs, rhs);

    else if (is_bitor2t(e))
      result = lhs | rhs;

    else if (is_bitand2t(e))
      result = lhs & rhs;
    else if (is_bitxor2t(e))
      result = lhs ^ rhs;

    else if (is_bitnand2t(e))
      result = wrapped_interval::bitnot(lhs & rhs);
    else if (is_bitnor2t(e))
      result = wrapped_interval::bitnot(lhs | rhs);
    else if (is_bitnxor2t(e))
      result = wrapped_interval::bitnot(lhs ^ rhs);
  }
  break;

  case expr2t::bitnot_id:
    result = wrapped_interval::bitnot(get_interval(to_bitnot2t(e).value));
    break;

  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
  case expr2t::equality_id:
  case expr2t::notequal_id:
  {
    const expr2tc &lhs = *e->get_sub_expr(0);
    const expr2tc &rhs = *e->get_sub_expr(1);

    auto lhs_i = get_interval(lhs);
    auto rhs_i = get_interval(rhs);

    if (is_equality2t(e))
      result = wrapped_interval::equality(lhs_i, rhs_i);

    else if (is_notequal2t(e))
      result = wrapped_interval::not_equal(lhs_i, rhs_i);

    else if (is_lessthan2t(e))
      result = wrapped_interval::less_than(lhs_i, rhs_i);

    else if (is_greaterthan2t(e))
      result = wrapped_interval::greater_than(lhs_i, rhs_i);

    else if (is_lessthanequal2t(e))
      result = wrapped_interval::less_than_equal(lhs_i, rhs_i);

    else if (is_greaterthanequal2t(e))
      result = wrapped_interval::greater_than_equal(lhs_i, rhs_i);

    break;
  }

  case expr2t::sideeffect_id:
    // This is probably a nondet
    log_debug("interval", "returning top for side effect {}", *e);
    break;

  default:
    log_debug("interval", "Couldn't compute interval for expr: {}", *e);
    break;
  }

  return result;
}

void goto_symext::make_bottom()
{
  log_status("Early exit");
}

void goto_symext::assume_rec(const expr2tc &cond, bool negation)
{
  if (is_comp_expr(cond))
  {
    assert(cond->get_num_sub_exprs() == 2);

    if (negation) // !x<y  ---> x>=y
    {
      if (is_lessthan2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::greaterthanequal_id,
          *cond->get_sub_expr(1));
      else if (is_lessthanequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::greaterthan_id,
          *cond->get_sub_expr(1));
      else if (is_greaterthan2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::lessthanequal_id,
          *cond->get_sub_expr(1));
      else if (is_greaterthanequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::lessthan_id, *cond->get_sub_expr(1));
      else if (is_equality2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::notequal_id, *cond->get_sub_expr(1));
      else if (is_notequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::equality_id, *cond->get_sub_expr(1));
    }
    else
      assume_rec(*cond->get_sub_expr(0), cond->expr_id, *cond->get_sub_expr(1));
  }
  else if (is_not2t(cond))
  {
    assume_rec(to_not2t(cond).value, !negation);
  }
  // de morgan
  else if (is_and2t(cond))
  {
    if (!negation)
      cond->foreach_operand([this](const expr2tc &e) { assume_rec(e, false); });
  }
  else if (is_or2t(cond))
  {
    if (negation)
      cond->foreach_operand([this](const expr2tc &e) { assume_rec(e, true); });
  }
  else if (is_symbol2t(cond))
  {
#if 0
    if (is_bv_type(cond) || is_bool_type(cond))
    {
      apply_assume_symbol_truth<wrapped_interval>(
          to_symbol2t(cond), negation);
    }
#endif
  }
  //added in case "cond = false" which happens when the ibex contractor results in empty set.
  else if (is_constant_bool2t(cond))
  {
    if ((negation && is_true(cond)) || (!negation && is_false(cond)))
    {
      make_bottom();
    }
  }
  else
    log_debug("interval", "[assume_rec] Missing support: {}", *cond);
}

void goto_symext::apply_assignment(expr2tc &lhs, expr2tc &rhs)
{
  // TODO: overflows
  const static bool overflow_mode =
    config.options.get_bool_option("overflow-check");
  if (overflow_mode)
    return;

  const bool lhs_precondition =
    (is_signedbv_type(lhs) || is_unsignedbv_type(lhs)) &&
    !symex_contains_unsupported(lhs) && is_symbol2t(lhs);
  const bool rhs_precondition =
    (is_signedbv_type(rhs) || is_unsignedbv_type(rhs)) &&
    !symex_contains_unsupported(rhs);
  if (lhs_precondition && rhs_precondition)
  {
    assert(is_symbol2t(lhs));
    update_symbol_interval(to_symbol2t(lhs), get_interval(rhs));
  }
}

void goto_symext::assume_rec(
  const expr2tc &lhs,
  expr2t::expr_ids id,
  const expr2tc &rhs)
{
  if (id == expr2t::equality_id)
  {
    assume_rec(lhs, expr2t::greaterthanequal_id, rhs);
    assume_rec(lhs, expr2t::lessthanequal_id, rhs);
    return;
  }

  if (id == expr2t::notequal_id)
    return; // won't do split

  if (id == expr2t::greaterthanequal_id)
    return assume_rec(rhs, expr2t::lessthanequal_id, lhs);

  if (id == expr2t::greaterthan_id)
    return assume_rec(rhs, expr2t::lessthan_id, lhs);

  if (id == expr2t::lessthan_id)
  {
    if (is_bv_type(lhs) && is_bv_type(rhs))
    {
      // TODO: To properly do this we need a way to invert functions
      if (!is_symbol2t(lhs))
      {
        // Gave-up for optimization
        auto new_lhs =
          add2tc(lhs->type, lhs, constant_int2tc(lhs->type, BigInt(1)));
        if (simplify(new_lhs))
          return assume_rec(new_lhs, expr2t::lessthanequal_id, rhs);
      }
      else if (!is_symbol2t(rhs))
      {
        // Gave-up for optimization
        auto new_rhs =
          sub2tc(rhs->type, rhs, constant_int2tc(rhs->type, BigInt(1)));
        if (simplify(new_rhs))
          return assume_rec(lhs, expr2t::lessthanequal_id, new_rhs);
      }
    }
    return assume_rec(lhs, expr2t::lessthanequal_id, rhs);
  }

  // we now have lhs <= rhs

  assert(id == expr2t::lessthanequal_id);

  if (is_bv_type(lhs) && is_bv_type(rhs))
    apply_assume_less(lhs, rhs);
}

void goto_symext::update_symbol_interval(
  const symbol2t &sym,
  const wrapped_interval value)
{
  intervals[sym.get_symbol_name()] = value;
}

void goto_symext::apply_assume_less(const expr2tc &a, const expr2tc &b)
{
  // 1. Apply contractor algorithms
  // 2. Update refs
  wrapped_interval lhs = get_interval(a);
  wrapped_interval rhs = get_interval(b);

  wrapped_interval s = lhs;
  s.make_le_than(rhs);
  wrapped_interval t = rhs;
  t.make_ge_than(lhs);

  // No need for widening, this is a restriction!
  if (is_symbol2t(a))
    update_symbol_interval(to_symbol2t(a), s);

  if (is_symbol2t(b))
    update_symbol_interval(to_symbol2t(b), t);
}

tvt goto_symext::assume_expression(const expr2tc &e)
{
  // TODO: overflows
  const static bool overflow_mode =
    config.options.get_bool_option("overflow-check");
  if (overflow_mode)
    return tvt(tvt::TV_UNKNOWN);

  if (symex_contains_unsupported(e))
    return tvt(tvt::TV_UNKNOWN);

  tvt result = eval_boolean_expression(e);

  if (result.is_false())
    return result;

  // TODO: we can't cut from the original symbol e.g.,
  //
  // int a = * ? 0 : 10 // a1: [0,10]
  // if(a < 5)
  //   ... a1: [0,4] is wrong. We need to maintain a set of guards over the symbol.
  // Or just move towards SSA as well

  //assume_rec(e);

  return result;
}
