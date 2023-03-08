/// \file
/// Interval Domain
// TODO: Ternary operators, lessthan into lessthanequal for integers
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <langapi/language_util.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>

// Let's start with all templates specializations.

template <>
integer_intervalt
interval_domaint::get_interval_from_symbol(const symbol2t &sym) const
{
  auto it = int_map.find(sym.thename);
  return it != int_map.end() ? int_map.at(sym.thename) : integer_intervalt();
}

template <>
real_intervalt
interval_domaint::get_interval_from_symbol(const symbol2t &sym) const
{
  auto it = real_map.find(sym.thename);
  return it != real_map.end() ? real_map.at(sym.thename) : real_intervalt();
}

template <>
void interval_domaint::update_symbol_interval(
  const symbol2t &sym,
  const integer_intervalt value)
{
  int_map[sym.thename] = value;
}

template <>
void interval_domaint::update_symbol_interval(
  const symbol2t &sym,
  const real_intervalt value)
{
  real_map[sym.thename] = value;
}

template <>
integer_intervalt interval_domaint::get_interval_from_const(const expr2tc &e)
{
  integer_intervalt result; // (-infinity, infinity)
  if(!is_constant_int2t(e) || !enable_modular_intervals)
    return result;
  auto value = to_constant_int2t(e).value;
  result.make_le_than(value);
  result.make_ge_than(value);
  return result;
}

template <>
real_intervalt interval_domaint::get_interval_from_const(const expr2tc &e)
{
  real_intervalt result; // (-infinity, infinity)
  if(!is_constant_floatbv2t(e) || !enable_modular_intervals)
    return result;

  auto value1 = to_constant_floatbv2t(e).value;
  auto value2 = to_constant_floatbv2t(e).value;

  if(value1.is_NaN() || value1.is_infinity())
    return result;

  value1.increment(true);
  value2.decrement(true);
  result.make_le_than(value1.to_double());
  result.make_ge_than(value2.to_double());
  assert(value2 <= value1);
  return result;
}

template <>
integer_intervalt
interval_domaint::generate_modular_interval<integer_intervalt>(
  const symbol2t sym) const
{
  auto t = sym.type;
  BigInt b;
  integer_intervalt result;
  if(is_unsignedbv_type(t))
  {
    b.setPower2(t->get_width());
    result.make_le_than(b);
    result.make_ge_than(0);
  }
  else if(is_signedbv_type(t))
  {
    b.setPower2(t->get_width() - 1);
    b = b + 1;
    result.make_le_than(b);
    result.make_ge_than(-b);
  }
  else
  {
    log_error("[interval] something bad happened");
    t->dump();
    abort();
  }
  return result;
}

template <>
real_intervalt interval_domaint::generate_modular_interval<real_intervalt>(
  const symbol2t) const
{
  // TODO: Support this
  real_intervalt t;
  return t;
}

template <class T>
void interval_domaint::apply_assignment(const expr2tc &lhs, const expr2tc &rhs)
{
  assert(is_symbol2t(lhs));
  // a = b
  auto a = generate_modular_interval<T>(to_symbol2t(lhs));
  auto b = get_interval<T>(rhs);

  T::contract_interval_le(a, b); // a <= b
  T::contract_interval_le(b, a); // b <= a

  if(fixpoint_counter[to_symbol2t(lhs).thename] >= delayed_widening_limit)
  {
if(widening_underaproximate_bound) {
    auto previous = get_interval_from_symbol<T>(to_symbol2t(lhs));
    auto upper_increased =
      a.upper_set && previous.upper_set && a.upper > previous.upper;
    auto lower_increased =
      a.lower_set && previous.lower_set && a.lower < previous.lower;
    if(upper_increased)
      a.upper_set = false;
    if(lower_increased)
      a.lower_set = false;
}
else {
    a.upper_set = false;
    a.lower_set = false;
}
  }
  update_symbol_interval(to_symbol2t(lhs), a);
}

template <class T>
T interval_domaint::get_interval(const expr2tc &e)
{
  log_debug("[interval] getting interval...");
  if(is_symbol2t(e))
    return get_interval_from_symbol<T>(to_symbol2t(e));

  if(is_neg2t(e))
    return -get_interval<T>(to_neg2t(e).value);

  // We do not care about overflows/overlaps for now
  if(is_typecast2t(e))
    return get_interval<T>(to_typecast2t(e).from);

  if(is_constant_number(e))
    return get_interval_from_const<T>(e);

  // Arithmetic?
  auto arith_op = std::dynamic_pointer_cast<arith_2ops>(e);
  auto ieee_arith_op = std::dynamic_pointer_cast<ieee_arith_2ops>(e);
  if(arith_op && enable_interval_arithmetic) // TODO: add support for float ops
  {
    // It should be safe to mix integers/floats in here.
    // The worst that can happen is an overaproximation
    auto lhs =
      get_interval<T>(arith_op ? arith_op->side_1 : ieee_arith_op->side_1);
    auto rhs =
      get_interval<T>(arith_op ? arith_op->side_2 : ieee_arith_op->side_2);

    if(is_add2t(e) || is_ieee_add2t(e))
      return lhs + rhs;

    if(is_sub2t(e) || is_ieee_sub2t(e))
      return lhs - rhs;

    if(is_mul2t(e) || is_ieee_mul2t(e))
      return lhs * rhs;

    if(is_div2t(e) || is_ieee_div2t(e))
      return lhs / rhs;

    // TODO: Add more as needed.
  }
  // We could not generate from the expr. Return top
  T result; // (-infinity, infinity)
  return result;
}

template <class Interval>
void interval_domaint::apply_assume_less(const expr2tc &a, const expr2tc &b)
{
  // 1. Apply contractor algorithms
  // 2. Update refs
  auto rhs = get_interval<Interval>(b);
  auto lhs = get_interval<Interval>(a);

  // TODO: Add less than equal
  Interval::contract_interval_le(lhs, rhs);
  if(is_symbol2t(a))
    update_symbol_interval(to_symbol2t(a), lhs);

  if(is_symbol2t(b))
    update_symbol_interval(to_symbol2t(b), rhs);

  if(rhs.is_bottom() || lhs.is_bottom())
    make_bottom();
}

template <>
bool interval_domaint::is_mapped<integer_intervalt>(const symbol2t &sym) const
{
  return int_map.find(sym.thename) == int_map.end();
}

template <>
bool interval_domaint::is_mapped<real_intervalt>(const symbol2t &sym) const
{
  return real_map.find(sym.thename) == real_map.end();
}

template <>
expr2tc interval_domaint::make_expression_value<integer_intervalt>(
  const integer_intervalt interval,
  const type2tc &type,
  bool upper) const
{
  return from_integer(upper ? interval.upper : interval.lower, type);
}

template <>
expr2tc interval_domaint::make_expression_value<real_intervalt>(
  const real_intervalt interval,
  const type2tc &type,
  bool upper) const
{
  constant_floatbv2tc value(ieee_floatt(ieee_float_spect(
    to_floatbv_type(type).fraction, to_floatbv_type(type).exponent)));

  const double d =
    (upper ? interval.upper : interval.lower).convert_to<double>();
  value->value.from_double(d);
  if(upper)
    value->value.increment(true);
  else
    value->value.decrement(true);

  return value;
}

template <class T>
expr2tc interval_domaint::make_expression_helper(const expr2tc &symbol) const
{
  symbol2t src = to_symbol2t(symbol);

  if(!is_mapped<T>(src))
    return gen_true_expr();

  const auto interval = get_interval_from_symbol<T>(src);
  if(interval.is_top())
    return gen_true_expr();

  if(interval.is_bottom())
    return gen_false_expr();

  std::vector<expr2tc> conjuncts;
  auto typecast = [&symbol](expr2tc v)
  {
    expr2tc new_expr = symbol;
    c_implicit_typecast_arithmetic(new_expr, v, *migrate_namespace_lookup);
    return new_expr;
  };
  if(interval.singleton())
  {
    expr2tc value = make_expression_value(interval, src.type, true);
    expr2tc new_expr = typecast(value);
    conjuncts.push_back(equality2tc(new_expr, value));
  }
  else
  {
    if(interval.upper_set)
    {
      expr2tc value = make_expression_value(interval, src.type, true);
      expr2tc new_expr = typecast(value);
      conjuncts.push_back(lessthanequal2tc(new_expr, value));
    }

    if(interval.lower_set)
    {
      expr2tc value = make_expression_value(interval, src.type, false);
      expr2tc new_expr = typecast(value);
      conjuncts.push_back(lessthanequal2tc(value, new_expr));
    }
  }
  return conjunction(conjuncts);
}

// END TEMPLATES

// TODO: refactor
void interval_domaint::output(std::ostream &out) const
{
  if(bottom)
  {
    out << "BOTTOM\n";
    return;
  }

  for(const auto &interval : int_map)
  {
    if(interval.second.is_top())
      continue;
    if(interval.second.lower_set)
      out << interval.second.lower << " <= ";
    out << interval.first;
    if(interval.second.upper_set)
      out << " <= " << interval.second.upper;
    out << "\n";
  }

  for(const auto &interval : real_map)
  {
    if(interval.second.is_top())
      continue;
    if(interval.second.lower_set)
      out << interval.second.lower << " <= ";
    out << interval.first;
    if(interval.second.upper_set)
      out << " <= " << interval.second.upper;
    out << "\n";
  }
}

void interval_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &ns)
{
  (void)ns;

  const goto_programt::instructiont &instruction = *from;
  switch(instruction.type)
  {
  case DECL:
    havoc_rec(instruction.code);
    break;

  case ASSIGN:
    assign(instruction.code);
    break;

  case GOTO:
  {
    // Comparing iterators is safe as the target must be within the same list
    // of instructions because this is a GOTO.
    goto_programt::const_targett next = from;
    next++;
    if(from->targets.front() != next) // If equal then a skip
    {
      if(next == to)
      {
        expr2tc guard = instruction.guard;
        make_not(guard);
        assume(guard);
      }
      else
        assume(instruction.guard);
    }
  }
  break;

  case ASSUME:
    assume(instruction.guard);
    break;

  case FUNCTION_CALL:
  {
    const code_function_call2t &code_function_call =
      to_code_function_call2t(instruction.code);
    if(!is_nil_expr(code_function_call.ret))
      havoc_rec(code_function_call.ret);
    break;
  }

  case ASSERT:
  {
    expr2tc code = instruction.code;
    ai_simplify(code, ns);
    break;
  }

  default:;
  }
}

bool interval_domaint::join(const interval_domaint &b)
{
  if(b.bottom)
    return false;
  if(bottom)
  {
    *this = b;
    return true;
  }

  bool result = false;

  auto f = [&result,this](auto& this_map, const auto& b_map, const auto& b_counter)
  {
    for(auto it = this_map.begin(); it != this_map.end();) // no it++
    {
      // search for the variable that needs to be merged
      // containers have different size and variable order
      const auto b_it = b_map.find(it->first);
      if(b_it == b_map.end())
      {
        it = this_map.erase(it);
        result = true;
      }
      else
      {
        auto test = b_counter.find(it->first);
        fixpoint_counter[it->first] =
          test == b_counter.end() ? 0 : test->second + 1;
        auto previous = it->second;
        it->second.join(b_it->second);
        if(it->second != previous)
          result = true;
        it++;
    }
    }
  };
  f(int_map, b.int_map, b.fixpoint_counter);
  f(real_map, b.real_map, b.fixpoint_counter);
  return result;
}

void interval_domaint::assign(const expr2tc &expr)
{
  assert(is_code_assign2t(expr));
  auto const &c = to_code_assign2t(expr);
  auto isbvop = is_bv_type(c.source) && is_bv_type(c.target);
  auto isfloatbvop = is_floatbv_type(c.source) && is_floatbv_type(c.target);
  auto ispointer = is_symbol2t(c.source);

  if(!is_symbol2t(c.target))
    return;
  if(isbvop)
    apply_assignment<integer_intervalt>(c.target, c.source);
  if(isfloatbvop)
    apply_assignment<real_intervalt>(c.target, c.source);
}

void interval_domaint::havoc_rec(const expr2tc &expr)
{
  if(is_if2t(expr))
  {
    havoc_rec(to_if2t(expr).true_value);
    havoc_rec(to_if2t(expr).false_value);
  }
  else if(is_typecast2t(expr))
  {
    havoc_rec(to_typecast2t(expr).from);
  }
  else if(is_symbol2t(expr) || is_code_decl2t(expr))
  {
    // Reset the interval domain if it is being reasigned (-infinity, +infinity).
    irep_idt identifier = is_symbol2t(expr) ? to_symbol2t(expr).thename
                                            : to_code_decl2t(expr).value;
    fixpoint_counter[identifier] = 0;
    if(is_bv_type(expr))
      int_map.erase(identifier);

    if(is_floatbv_type(expr))
      real_map.erase(identifier);
  }
  else
    log_debug("[havoc_rec] Missing support: {}", *expr);
}

void interval_domaint::assume_rec(
  const expr2tc &lhs,
  expr2t::expr_ids id,
  const expr2tc &rhs)
{
  if(is_typecast2t(lhs))
    return assume_rec(to_typecast2t(lhs).from, id, rhs);

  if(is_typecast2t(rhs))
    return assume_rec(lhs, id, to_typecast2t(rhs).from);

  if(id == expr2t::equality_id)
  {
    assume_rec(lhs, expr2t::greaterthanequal_id, rhs);
    assume_rec(lhs, expr2t::lessthanequal_id, rhs);
    return;
  }

  if(id == expr2t::notequal_id)
    return; // won't do split

  if(id == expr2t::greaterthanequal_id)
    return assume_rec(rhs, expr2t::lessthanequal_id, lhs);

  if(id == expr2t::greaterthan_id)
    return assume_rec(rhs, expr2t::lessthan_id, lhs);

  // we now have lhs <  rhs or
  //             lhs <= rhs

  assert(id == expr2t::lessthan_id || id == expr2t::lessthanequal_id);

  if(is_bv_type(lhs) && is_bv_type(rhs))
    apply_assume_less<integer_intervalt>(lhs, rhs);
  else if(is_floatbv_type(lhs) && is_floatbv_type(rhs))
    apply_assume_less<real_intervalt>(lhs, rhs);
}

void interval_domaint::assume(const expr2tc &cond)
{
  expr2tc new_cond = cond;
  simplify(new_cond);
  assume_rec(new_cond, false);
}

void interval_domaint::assume_rec(const expr2tc &cond, bool negation)
{
  if(is_comp_expr(cond))
  {
    assert(cond->get_num_sub_exprs() == 2);

    if(negation) // !x<y  ---> x>=y
    {
      if(is_lessthan2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::greaterthanequal_id,
          *cond->get_sub_expr(1));
      else if(is_lessthanequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::greaterthan_id,
          *cond->get_sub_expr(1));
      else if(is_greaterthan2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::lessthanequal_id,
          *cond->get_sub_expr(1));
      else if(is_greaterthanequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::lessthan_id, *cond->get_sub_expr(1));
      else if(is_equality2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::notequal_id, *cond->get_sub_expr(1));
      else if(is_notequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::equality_id, *cond->get_sub_expr(1));
    }
    else
      assume_rec(*cond->get_sub_expr(0), cond->expr_id, *cond->get_sub_expr(1));
  }
  else if(is_not2t(cond))
  {
    assume_rec(to_not2t(cond).value, !negation);
  }
  // de morgan
  else if(is_and2t(cond))
  {
    if(!negation)
      cond->foreach_operand([this](const expr2tc &e) { assume_rec(e, false); });
  }
  else if(is_or2t(cond))
  {
    if(negation)
      cond->foreach_operand([this](const expr2tc &e) { assume_rec(e, true); });
  }
  else
    log_debug("[assume_rec] Missing support: {}", *cond);
}

void interval_domaint::dump() const
{
  std::ostringstream oss;
  output(oss);
  log_debug("{}", oss.str());
}
expr2tc interval_domaint::make_expression(const expr2tc &symbol) const
{
  assert(is_symbol2t(symbol));
  if(is_bv_type(symbol))
    return make_expression_helper<integer_intervalt>(symbol);
  if(is_floatbv_type(symbol))
    return make_expression_helper<real_intervalt>(symbol);
  return gen_true_expr();
}

// TODO: simplify
bool interval_domaint::ai_simplify(expr2tc &condition, const namespacet &ns)
  const
{
  (void)ns;

  if(!enable_assertion_simplification)
    return true;

  bool unchanged = true;
  interval_domaint d(*this);

  // merge intervals to properly handle conjunction
  if(is_and2t(condition)) // May be directly representable
  {
    // TODO: This is not working, reimplement this using other logic
    log_debug("[interval] Conjuction");
    interval_domaint a;
    a.make_top();        // a is everything
    a.assume(condition); // Restrict a to an over-approximation
                         //  of when condition is true
    if(!a.join(d))       // If d (this) is included in a...
    {                    // Then the condition is always true
      unchanged = is_true(condition);
      condition = gen_true_expr();
    }
  }
  else if(is_symbol2t(condition))
  {
    // TODO: we have to handle symbol expression
  }
  else // Less likely to be representable
  {
    log_debug("[interval] not");
    expr2tc not_condition = condition;
    make_not(not_condition);
    d.assume(not_condition); // Restrict to when condition is false
    if(d.is_bottom())        // If there there are none...
    {                        // Then the condition is always true
      unchanged = is_true(condition);
      condition = gen_true_expr();
    }
  }

  return unchanged;
}
