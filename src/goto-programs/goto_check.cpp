/*******************************************************************
 Module: GOTO Programs

 Author: Daniel Kroening, kroening@kroening.com Lucas Cordeiro,
 lcc08r@ecs.soton.ac.uk

 \*******************************************************************/

#include <goto-programs/goto_check.h>
#include <util/arith_tools.h>
#include <util/array_name.h>
#include <util/base_type.h>
#include <util/expr_util.h>
#include <util/guard.h>
#include <util/i2string.h>
#include <util/location.h>
#include <util/simplify_expr.h>

class goto_checkt
{
public:
  goto_checkt(const namespacet &_ns, optionst &_options)
    : ns(_ns),
      options(_options),
      disable_bounds_check(options.get_bool_option("no-bounds-check")),
      disable_pointer_check(options.get_bool_option("no-pointer-check")),
      disable_div_by_zero_check(
        options.get_bool_option("no-div-by-zero-check")),
      enable_overflow_check(options.get_bool_option("overflow-check")),
      enable_nan_check(options.get_bool_option("nan-check"))
  {
  }

  void goto_check(goto_programt &goto_program);

protected:
  const namespacet &ns;
  optionst &options;

  void check(const expr2tc &expr, const locationt &location);

  void check_rec(
    const expr2tc &expr,
    guardt &guard,
    const locationt &loc,
    bool address);

  void div_by_zero_check(
    const expr2tc &expr,
    const guardt &guard,
    const locationt &loc);

  void
  bounds_check(const expr2tc &expr, const guardt &guard, const locationt &loc);

  void pointer_rel_check(
    const expr2tc &expr,
    const guardt &guard,
    const locationt &loc);

  void overflow_check(
    const expr2tc &expr,
    const guardt &guard,
    const locationt &loc);

  void float_overflow_check(
    const expr2tc &expr,
    const guardt &guard,
    const locationt &loc);

  void
  nan_check(const expr2tc &expr, const guardt &guard, const locationt &loc);

  void add_guarded_claim(
    const expr2tc &expr,
    const std::string &comment,
    const std::string &property,
    const locationt &location,
    const guardt &guard);

  goto_programt new_code;
  std::set<expr2tc> assertions;

  bool disable_bounds_check;
  bool disable_pointer_check;
  bool disable_div_by_zero_check;
  bool enable_overflow_check;
  bool enable_nan_check;
};

void goto_checkt::div_by_zero_check(
  const expr2tc &expr,
  const guardt &guard,
  const locationt &loc)
{
  if(disable_div_by_zero_check)
    return;

  assert(is_div2t(expr) || is_modulus2t(expr));

  // add divison by zero subgoal
  expr2tc side_2;
  if(is_div2t(expr))
    side_2 = to_div2t(expr).side_2;
  else
    side_2 = to_modulus2t(expr).side_2;

  expr2tc zero = gen_zero(side_2->type);
  assert(!is_nil_expr(zero));

  add_guarded_claim(
    notequal2tc(side_2, zero),
    "division by zero",
    "division-by-zero",
    loc,
    guard);
}

void goto_checkt::float_overflow_check(
  const expr2tc &expr,
  const guardt &guard,
  const locationt &loc)
{
  if(!enable_overflow_check)
    return;

  // First, check type.
  const type2tc &type = ns.follow(expr->type);
  if(!is_floatbv_type(type))
    return;

  // We could use get_sub_expr(idx) instead of this switch case,
  // but the documentation states that the order is not guaranteed
  expr2tc side_1, side_2;
  switch(expr->expr_id)
  {
  case expr2t::ieee_add_id:
    side_1 = to_ieee_add2t(expr).side_1;
    side_2 = to_ieee_add2t(expr).side_2;
    break;

  case expr2t::ieee_sub_id:
    side_1 = to_ieee_sub2t(expr).side_1;
    side_2 = to_ieee_sub2t(expr).side_2;
    break;

  case expr2t::ieee_mul_id:
    side_1 = to_ieee_mul2t(expr).side_1;
    side_2 = to_ieee_mul2t(expr).side_2;
    break;

  case expr2t::ieee_div_id:
    side_1 = to_ieee_div2t(expr).side_1;
    side_2 = to_ieee_div2t(expr).side_2;
    break;

  default:
    abort();
  }

  if(is_ieee_div2t(expr))
  {
    // Can overflow if dividing by something small
    expr2tc op0_inf = expr2tc(new isinf2t(side_1));
    expr2tc new_inf = expr2tc(new isinf2t(expr));
    make_not(new_inf);

    or2tc overflow_check(op0_inf, new_inf);

    add_guarded_claim(
      overflow_check,
      "arithmetic overflow on floating-point " + get_expr_id(expr),
      "overflow",
      loc,
      guard);
  }
  else if(is_ieee_add2t(expr) || is_ieee_sub2t(expr) || is_ieee_mul2t(expr))
  {
    // Can overflow
    expr2tc op0_inf = expr2tc(new isinf2t(side_1));
    expr2tc op1_inf = expr2tc(new isinf2t(side_2));
    or2tc operands_or(op0_inf, op1_inf);

    expr2tc new_inf = expr2tc(new isinf2t(expr));
    make_not(new_inf);

    or2tc overflow_check(operands_or, new_inf);

    add_guarded_claim(
      overflow_check,
      "arithmetic overflow on floating-point " + get_expr_id(expr),
      "overflow",
      loc,
      guard);
  }
}

void goto_checkt::overflow_check(
  const expr2tc &expr,
  const guardt &guard,
  const locationt &loc)
{
  if(!enable_overflow_check)
    return;

  // First, check type.
  const type2tc &type = ns.follow(expr->type);
  if(!is_signedbv_type(type))
    return;

  // Don't check pointer overflow
  if(is_pointer_type(*expr->get_sub_expr(0)))
    return;

  // add overflow subgoal
  expr2tc overflow = is_neg2t(expr)
                       ? expr2tc(new overflow_neg2t(to_neg2t(expr).value))
                       : expr2tc(new overflow2t(expr));
  make_not(overflow);

  add_guarded_claim(
    overflow,
    "arithmetic overflow on " + get_expr_id(expr),
    "overflow",
    loc,
    guard);
}

void goto_checkt::nan_check(
  const expr2tc &expr,
  const guardt &guard,
  const locationt &loc)
{
  if(!enable_nan_check)
    return;

  // First, check type.
  const type2tc &type = ns.follow(expr->type);
  if(!is_floatbv_type(type))
    return;

  // add nan subgoal
  expr2tc isnan = expr2tc(new isnan2t(expr));
  make_not(isnan);

  add_guarded_claim(isnan, "NaN on " + get_expr_id(expr), "NaN", loc, guard);
}

void goto_checkt::pointer_rel_check(
  const expr2tc &expr,
  const guardt &guard,
  const locationt &loc)
{
  if(disable_pointer_check)
    return;

  assert(expr->get_num_sub_exprs() == 2);

  if(
    is_pointer_type(*expr->get_sub_expr(0)) &&
    is_pointer_type(*expr->get_sub_expr(1)))
  {
    // add same-object subgoal
    expr2tc side_1 = *expr->get_sub_expr(0);
    expr2tc side_2 = *expr->get_sub_expr(1);

    same_object2tc same_object(side_1, side_2);
    add_guarded_claim(
      same_object, "Same object violation", "pointer", loc, guard);
  }
}

static bool has_dereference(const expr2tc &expr)
{
  if(is_dereference2t(expr))
    return true;

  if(is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value))
    // This is an index of a pointer, which is a dereference
    return true;

  // Recurse through all subsequent source objects, which are always operand
  // zero.
  bool found = false;
  expr->foreach_operand(
    [&found](const expr2tc &e) { found |= has_dereference(e); });

  return found;
}

void goto_checkt::bounds_check(
  const expr2tc &expr,
  const guardt &guard,
  const locationt &loc)
{
  (void)guard;
  (void)loc;

  if(disable_bounds_check)
    return;

  if(!is_index2t(expr))
    return;

  index2t ind = to_index2t(expr);

  // Don't bounds check the initial index of argv in the "main" function; it's
  // always correct, and just adds needless claims. In the past a "no bounds
  // check" attribute in old irep handled this.
  if(
    is_symbol2t(ind.source_value) &&
    to_symbol2t(ind.source_value).thename == "argv'" &&
    is_symbol2t(ind.index) && to_symbol2t(ind.index).thename == "argc'")
    return;

  if(
    is_symbol2t(ind.source_value) &&
    to_symbol2t(ind.source_value).thename == "envp'" &&
    is_symbol2t(ind.index) && to_symbol2t(ind.index).thename == "envp_size'")
    return;

  const type2tc &t = ns.follow(ind.source_value->type);
  if(is_pointer_type(t))
    return; // done by the pointer code

  // Otherwise, if there's a dereference in the array source, this bounds check
  // should be performed by the symex-time dereferencing code, as the base thing
  // being accessed may be anything.
  if(has_dereference(ind.source_value))
    return;

  std::string name =
    "array bounds violated: " + array_name(ns, ind.source_value);
  const expr2tc &the_index = ind.index;

  // Lower bound access should be greather than zero
  expr2tc zero = gen_zero(the_index->type);
  assert(!is_nil_expr(zero));

  greaterthanequal2tc lower(the_index, zero);
  add_guarded_claim(lower, name + " lower bound", "array bounds", loc, guard);

  assert(is_array_type(t) || is_string_type(t));

  // We can't check the upper bound of an infinite sized array
  if(is_array_type(t) && to_array_type(t).size_is_infinite)
    return;

  const expr2tc &array_size =
    is_array_type(t)
      ? to_array_type(t).array_size
      : constant_int2tc(get_uint64_type(), to_string_type(t).get_length());

  // Cast size to index type
  typecast2tc casted_size(the_index->type, array_size);
  lessthan2tc upper(the_index, casted_size);
  add_guarded_claim(upper, name + " upper bound", "array bounds", loc, guard);
}

void goto_checkt::add_guarded_claim(
  const expr2tc &expr,
  const std::string &comment,
  const std::string &property,
  const locationt &location,
  const guardt &guard)
{
  expr2tc e = expr;

  // first try simplifier on it
  base_type(e, ns);
  simplify(e);

  if(!options.get_bool_option("all-claims") && is_true(e))
    return;

  // add the guard
  expr2tc new_expr = guard.is_true() ? e : implies2tc(guard.as_expr(), e);

  // Check if we're not adding the same assertion twice
  if(assertions.insert(new_expr).second)
  {
    goto_programt::targett t = new_code.add_instruction(ASSERT);
    t->guard = new_expr;
    t->location = location;
    t->location.comment(comment);
    t->location.property(property);
  }
}

void goto_checkt::check_rec(
  const expr2tc &expr,
  guardt &guard,
  const locationt &loc,
  bool address)
{
  if(is_nil_expr(expr))
    return;

  if(address)
  {
    switch(expr->expr_id)
    {
    case expr2t::dereference_id:
      check_rec(to_dereference2t(expr).value, guard, loc, false);
      break;

    case expr2t::index_id:
      check_rec(to_index2t(expr).source_value, guard, loc, true);
      check_rec(to_index2t(expr).index, guard, loc, false);
      break;

    default:
      expr->foreach_operand([this, &guard, &loc](const expr2tc &e) {
        check_rec(e, guard, loc, true);
      });
    }

    return;
  }

  switch(expr->expr_id)
  {
  case expr2t::address_of_id:
    check_rec(to_address_of2t(expr).ptr_obj, guard, loc, true);
    return;

  case expr2t::and_id:
  case expr2t::or_id:
  {
    assert(is_bool_type(expr));

    guardt old_guards(guard);

    bool is_or = is_or2t(expr);
    expr->foreach_operand([this, &is_or, &guard, &loc](const expr2tc &e) {
      assert(is_bool_type(e));
      check_rec(e, guard, loc, false);

      if(is_or)
      {
        expr2tc tmp = e;
        make_not(tmp);
        guard.add(tmp);
      }
      else
        guard.add(e);
    });

    guard.swap(old_guards);
    return;
  }

  case expr2t::if_id:
  {
    auto i = to_if2t(expr);
    assert(is_bool_type(i.cond));

    // Check cond
    check_rec(i.cond, guard, loc, false);

    // Check true path
    {
      guardt old_guards(guard);
      guard.add(i.cond);
      check_rec(i.true_value, guard, loc, false);
      guard.swap(old_guards);
    }

    // Check false path, the guard is negated
    {
      guardt old_guards(guard);
      expr2tc tmp = i.cond;
      make_not(tmp);
      guard.add(tmp);
      check_rec(i.false_value, guard, loc, false);
      guard.swap(old_guards);
    }

    return;
  }

  default:
    break;
  }

  expr->foreach_operand([this, &guard, &loc](const expr2tc &e) {
    check_rec(e, guard, loc, false);
  });

  switch(expr->expr_id)
  {
  case expr2t::index_id:
    bounds_check(expr, guard, loc);
    return;

  case expr2t::div_id:
  case expr2t::modulus_id:
    div_by_zero_check(expr, guard, loc);
    /* fallthrough */

  case expr2t::shl_id:
  case expr2t::neg_id:
  case expr2t::add_id:
  case expr2t::sub_id:
  case expr2t::mul_id:
  {
    overflow_check(expr, guard, loc);
    break;
  }

  case expr2t::ieee_add_id:
  case expr2t::ieee_sub_id:
  case expr2t::ieee_mul_id:
  case expr2t::ieee_div_id:
  {
    // No division by zero for ieee_div, as it's defined behaviour
    float_overflow_check(expr, guard, loc);
    nan_check(expr, guard, loc);
    break;
  }

  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
    pointer_rel_check(expr, guard, loc);
    break;

  default:
    break;
  }
}

void goto_checkt::check(const expr2tc &expr, const locationt &loc)
{
  guardt guard;
  check_rec(expr, guard, loc, false);
}

void goto_checkt::goto_check(goto_programt &goto_program)
{
  // Not a ranged loop because we need it to be an iterator :/
  for(goto_programt::instructionst::iterator it =
        goto_program.instructions.begin();
      it != goto_program.instructions.end();
      it++)
  {
    goto_programt::instructiont &i = *it;
    const locationt &loc = i.location;

    new_code.clear();
    assertions.clear();

    check(i.guard, loc);

    if(i.is_other())
    {
      if(is_code_expression2t(i.code))
      {
        check(i.code, loc);
      }
      else if(is_code_printf2t(i.code))
      {
        i.code->foreach_operand(
          [this, &loc](const expr2tc &e) { check(e, loc); });
      }
    }
    else if(i.is_assign())
    {
      const code_assign2t &assign = to_code_assign2t(i.code);
      check(assign.target, loc);
      check(assign.source, loc);
    }
    else if(i.is_function_call())
    {
      i.code->foreach_operand(
        [this, &loc](const expr2tc &e) { check(e, loc); });
    }
    else if(i.is_return())
    {
      const code_return2t &ret = to_code_return2t(i.code);
      check(ret.operand, loc);
    }

    // insert new instructions -- make sure targets are not moved
    while(!new_code.instructions.empty())
    {
      goto_program.insert_swap(it, new_code.instructions.front());
      new_code.instructions.pop_front();
      it++;
    }
  }
}

void goto_check(
  const namespacet &ns,
  optionst &options,
  goto_programt &goto_program)
{
  goto_checkt goto_check(ns, options);
  goto_check.goto_check(goto_program);
}

void goto_check(
  const namespacet &ns,
  optionst &options,
  goto_functionst &goto_functions)
{
  goto_checkt goto_check(ns, options);

  for(auto &it : goto_functions.function_map)
  {
    if(!it.second.body.empty())
      goto_check.goto_check(it.second.body);
  }
}
