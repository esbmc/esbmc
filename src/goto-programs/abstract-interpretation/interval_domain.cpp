/// \file
/// Interval Domain
// TODO: Ternary operators, lessthan into lessthanequal for integers
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <goto-programs/abstract-interpretation/bitwise_bounds.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/std_expr.h>
#ifdef ENABLE_GOTO_CONTRACTOR
#include <goto-programs/goto_contractor.h>
#endif
// Let's start with all templates specializations.

template <>
integer_intervalt
interval_domaint::get_interval_from_symbol(const symbol2t &sym) const
{
  auto it = int_map.find(sym.thename);
  return it != int_map.end() ? it->second : integer_intervalt();
}

template <>
real_intervalt
interval_domaint::get_interval_from_symbol(const symbol2t &sym) const
{
  auto it = real_map.find(sym.thename);
  return it != real_map.end() ? it->second : real_intervalt();
}

template <>
wrapped_interval
interval_domaint::get_interval_from_symbol(const symbol2t &sym) const
{
  auto it = wrap_map.find(sym.thename);
  return it != wrap_map.end() ? it->second : wrapped_interval(sym.type);
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
void interval_domaint::update_symbol_interval(
  const symbol2t &sym,
  const wrapped_interval value)
{
  wrap_map[sym.thename] = value;
}

template <class Interval>
void interval_domaint::apply_assume_symbol_truth(
  const symbol2t &sym,
  bool is_false)
{
  Interval interval = get_interval_from_symbol<Interval>(sym);
  // [0,0]
  if (is_false)
  {
    interval.set_upper(0);
    interval.set_lower(0);
  }
  // [1, infinity]
  else if (is_unsignedbv_type(sym.type) || is_bool_type(sym.type))
    interval.make_ge_than(1);
  else if (is_signedbv_type(sym.type))
  {
    if (interval.lower && interval.get_lower() == 0)
      interval.make_ge_than(1);
    else if (interval.upper && interval.get_upper() == 0)
      interval.make_le_than(-1);
  }

  update_symbol_interval(sym, interval);
}

template <>
integer_intervalt
interval_domaint::get_interval_from_const(const expr2tc &e) const
{
  integer_intervalt result; // (-infinity, infinity)
  if (!is_constant_int2t(e))
    return result;
  auto value = to_constant_int2t(e).value;
  result.make_le_than(value);
  result.make_ge_than(value);
  return result;
}

#include <cmath>

template <>
real_intervalt interval_domaint::get_interval_from_const(const expr2tc &e) const
{
  real_intervalt result; // (-infinity, infinity)
  if (!is_constant_floatbv2t(e))
    return result;

  auto real_value = to_constant_floatbv2t(e).value;

  // Health check, is the convertion to double ok? See #1037
  if (!real_value.is_normal() || real_value.is_zero())
  {
    if (real_value.is_double())
      log_warning("ESBMC fails to convert {} into double", *e);

    // Give up for top!
    return result;
  }

  auto value1 = to_constant_floatbv2t(e).value;
  auto value2 = to_constant_floatbv2t(e).value;
  value1.increment(true);
  value2.decrement(true);

  if (
    value1.is_NaN() || value1.is_infinity() || value2.is_NaN() ||
    value2.is_infinity())
  {
    assert(result.is_top() && !result.is_bottom());
    return result;
  }

  // [value2, value1]
  // a <= value1
  if (value1.is_double())
    result.make_le_than(value1.to_double());
  else
    log_warning("Failed to convert value1: {}", value1.to_string_decimal(10));

  // a >= value2
  if (value2.is_double())
    result.make_ge_than(value2.to_double());
  else
    log_warning(
      "Failed to convert value2: {}", value2.to_string_decimal(10)); //

  assert(!result.is_bottom());
  return result;
}

template <>
wrapped_interval
interval_domaint::get_interval_from_const(const expr2tc &e) const
{
  wrapped_interval result(e->type); // [0, 2^(length(e->type)) - 1]
  if (!is_constant_int2t(e))
    return result;
  auto value = to_constant_int2t(e).value;
  result.set_lower(value);
  result.set_upper(value);
  assert(!result.is_bottom());
  return result;
}

template <>
integer_intervalt
interval_domaint::generate_modular_interval<integer_intervalt>(
  const symbol2t sym) const
{
  auto t = sym.type;
  integer_intervalt result;
  if (is_unsignedbv_type(t))
  {
    result.make_le_than(BigInt::power2m1(t->get_width()));
    result.make_ge_than(0);
  }
  else if (is_signedbv_type(t))
  {
    BigInt b = BigInt::power2(t->get_width() - 1);
    result.make_ge_than(-b);
    result.make_le_than(b - 1);
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
  assert(t.is_top() && !t.is_bottom());
  return t;
}

template <>
wrapped_interval interval_domaint::generate_modular_interval<wrapped_interval>(
  const symbol2t sym) const
{
  // Wrapped intervals are modular by definition
  wrapped_interval t(sym.type);
  return t;
}

template <class T>
void interval_domaint::apply_assignment(
  const expr2tc &lhs,
  const expr2tc &rhs,
  bool recursive)
{
  assert(is_symbol2t(lhs));
  const symbol2t &sym = to_symbol2t(lhs);

  // a = b
  auto b = get_interval<T>(rhs);
  if (enable_modular_intervals)
  {
    auto a = generate_modular_interval<T>(sym);
    b.intersect_with(a);
  }

  if (recursive)
  {
    auto previous = get_interval_from_symbol<T>(sym);
    b.join(previous);
  }

  // TODO: add classic algorithm
  update_symbol_interval(sym, b);
}

template <class T>
T interval_domaint::extrapolate_intervals(const T &before, const T &after)
{
  T result; // Full extrapolation
  bool lower_decreased = !after.lower || (before.lower && after.lower &&
                                          *after.lower < *before.lower);
  bool upper_increased = !after.upper || (before.upper && after.upper &&
                                          *after.upper > *before.upper);

  if ((lower_decreased || upper_increased) && !widening_under_approximate_bound)
    return result;

  // Set lower bound: if we didn't decrease then just update the interval
  if (!lower_decreased)
    result.lower = before.lower;

  // Set upper bound:
  if (!upper_increased)
    result.upper = before.upper;
  return result;
}

template <>
wrapped_interval interval_domaint::extrapolate_intervals(
  const wrapped_interval &before,
  const wrapped_interval &after)
{
  return wrapped_interval::extrapolate_to(before, after);
}

template <class Interval>
Interval interval_domaint::get_top_interval_from_expr(const expr2tc &) const
{
  Interval result;
  return result;
}

template <>
wrapped_interval
interval_domaint::get_top_interval_from_expr(const expr2tc &e) const
{
  wrapped_interval result(e->type);
  return result;
}

template <class T>
T interval_domaint::interpolate_intervals(const T &before, const T &after)
{
  T result;

  // before: [-infinity, +infinity], after: [a,b] ==> [a,b]
  bool lower_increased = !before.lower && after.lower;
  bool upper_decreased = !before.upper && after.upper;

  if (lower_increased)
    result.lower = after.lower;

  if (upper_decreased)
    result.upper = after.upper;
  return result;
}

template <class T>
T interval_domaint::get_interval(const expr2tc &e) const
{
  T result = get_top_interval_from_expr<T>(e);

  // TODO: I probably can refactor these in a better way
  switch (e->expr_id)
  {
  case expr2t::constant_bool_id:
    result.set_lower(to_constant_bool2t(e).is_true());
    result.set_upper(to_constant_bool2t(e).is_true());
    break;

  case expr2t::constant_int_id:
  case expr2t::constant_fixedbv_id:
  case expr2t::constant_floatbv_id:
    result = get_interval_from_const<T>(e);
    break;

  case expr2t::symbol_id:
    result = get_interval_from_symbol<T>(to_symbol2t(e));
    break;

  case expr2t::neg_id:
    result = -get_interval<T>(to_neg2t(e).value);
    break;

  case expr2t::not_id:
    result = T::invert_bool(get_interval<T>(to_not2t(e).value));
    break;

  case expr2t::or_id:
  case expr2t::and_id:
  case expr2t::xor_id:
  case expr2t::implies_id:
  {
    const auto &logic_op = dynamic_cast<const logic_2ops &>(*e);
    tvt lhs = eval_boolean_expression(logic_op.side_1, *this);
    tvt rhs = eval_boolean_expression(logic_op.side_2, *this);

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
    auto cond = get_interval<T>(to_if2t(e).cond);
    auto lhs = get_interval<T>(to_if2t(e).true_value);
    auto rhs = get_interval<T>(to_if2t(e).false_value);
    result = T::ternary_if(cond, lhs, rhs);
    break;
  }

  case expr2t::typecast_id:
  {
    // Special case: boolean
    if (is_bool_type(to_typecast2t(e).type))
    {
      tvt truth = eval_boolean_expression(to_typecast2t(e).from, *this);
      result.set_lower(0);
      result.set_upper(1);

      if (truth.is_true())
        result.set_lower(1);

      if (truth.is_false())
        result.set_upper(0);

      break;
    }
    auto inner = get_interval<T>(to_typecast2t(e).from);
    result = T::cast(inner, to_typecast2t(e).type);
    break;
  }

  case expr2t::add_id:
  case expr2t::sub_id:
  case expr2t::mul_id:
  case expr2t::div_id:
  case expr2t::modulus_id:
    if (enable_interval_arithmetic)
    {
      const auto &arith_op = dynamic_cast<const arith_2ops &>(*e);
      auto lhs = get_interval<T>(arith_op.side_1);
      auto rhs = get_interval<T>(arith_op.side_2);

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
    if (enable_interval_bitwise_arithmetic)
    {
      const auto &bit_op = dynamic_cast<const bit_2ops &>(*e);
      auto lhs = get_interval<T>(bit_op.side_1);
      auto rhs = get_interval<T>(bit_op.side_2);
      lhs.type = bit_op.side_1->type;
      rhs.type = bit_op.side_2->type;
      if (is_shl2t(e))
        result = T::left_shift(lhs, rhs);

      else if (is_ashr2t(e))
        result = T::arithmetic_right_shift(lhs, rhs);

      else if (is_lshr2t(e))
        result = T::logical_right_shift(lhs, rhs);

      else if (is_bitor2t(e))
        result = lhs | rhs;

      else if (is_bitand2t(e))
        result = lhs & rhs;
      else if (is_bitxor2t(e))
        result = lhs ^ rhs;

      else if (is_bitnand2t(e))
        result = T::bitnot(lhs & rhs);
      else if (is_bitnor2t(e))
        result = T::bitnot(lhs | rhs);
      else if (is_bitnxor2t(e))
        result = T::bitnot(lhs ^ rhs);
    }
    break;

  case expr2t::bitnot_id:
    if (enable_interval_bitwise_arithmetic)
      result = T::bitnot(get_interval<T>(to_bitnot2t(e).value));
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

    auto lhs_i = get_interval<T>(lhs);
    auto rhs_i = get_interval<T>(rhs);

    if (is_equality2t(e))
      result = T::equality(lhs_i, rhs_i);

    else if (is_notequal2t(e))
      result = T::not_equal(lhs_i, rhs_i);

    else if (is_lessthan2t(e))
      result = T::less_than(lhs_i, rhs_i);

    else if (is_greaterthan2t(e))
      result = T::greater_than(lhs_i, rhs_i);

    else if (is_lessthanequal2t(e))
      result = T::less_than_equal(lhs_i, rhs_i);

    else if (is_greaterthanequal2t(e))
      result = T::greater_than_equal(lhs_i, rhs_i);

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

template <class Interval>
void interval_domaint::apply_assume_less(const expr2tc &a, const expr2tc &b)
{
  // 1. Apply contractor algorithms
  // 2. Update refs
  auto lhs = get_interval<Interval>(a);
  auto rhs = get_interval<Interval>(b);

  // TODO: Add less than equal
  if (enable_contraction_for_abstract_states)
    Interval::contract_interval_le(lhs, rhs);
  else
  {
    if (is_symbol2t(a) && is_symbol2t(b))
      lhs.make_sound_le(rhs);
    else
    {
      if (rhs.upper)
        lhs.make_le_than(rhs.get_upper());

      if (lhs.lower)
        rhs.make_ge_than(lhs.get_lower());
    }
  }
  // No need for widening, this is a restriction!
  if (is_symbol2t(a))
    update_symbol_interval<Interval>(to_symbol2t(a), lhs);

  if (is_symbol2t(b))
    update_symbol_interval<Interval>(to_symbol2t(b), rhs);

  if (rhs.is_bottom() || lhs.is_bottom())
    make_bottom();
}

template <>
void interval_domaint::apply_assume_less<wrapped_interval>(
  const expr2tc &a,
  const expr2tc &b)
{
  // 1. Apply contractor algorithms
  // 2. Update refs
  auto lhs = get_interval<wrapped_interval>(a);
  auto rhs = get_interval<wrapped_interval>(b);

  auto s = lhs;
  s.make_le_than(rhs);
  auto t = rhs;
  t.make_ge_than(lhs);

  // No need for widening, this is a restriction!
  if (is_symbol2t(a))
    update_symbol_interval(to_symbol2t(a), s);

  if (is_symbol2t(b))
    update_symbol_interval(to_symbol2t(b), t);

  if (s.is_bottom() || t.is_bottom())
    make_bottom();
}

template <>
bool interval_domaint::is_mapped<integer_intervalt>(const symbol2t &sym) const
{
  return int_map.find(sym.thename) != int_map.end();
}

template <>
bool interval_domaint::is_mapped<real_intervalt>(const symbol2t &sym) const
{
  return real_map.find(sym.thename) != real_map.end();
}

template <>
bool interval_domaint::is_mapped<wrapped_interval>(const symbol2t &sym) const
{
  return wrap_map.find(sym.thename) != wrap_map.end();
}

template <>
expr2tc interval_domaint::make_expression_value<integer_intervalt>(
  const integer_intervalt &interval,
  const type2tc &type,
  bool is_upper) const
{
  return from_integer(is_upper ? *interval.upper : *interval.lower, type);
}

template <>
expr2tc interval_domaint::make_expression_value<real_intervalt>(
  const real_intervalt &interval,
  const type2tc &type,
  bool upper) const
{
  expr2tc value = gen_zero(type);
  constant_floatbv2t &v = to_constant_floatbv2t(value);

  const auto d =
    (upper ? *interval.upper : *interval.lower).convert_to<double>();
  v.value.from_double(d);

  // 'from_double' changes the original spec. This makes solvers complain that we are comparing
  // 'orange' floats to 'apple' floats. To fix this, we need to convert the spec back.
  const ieee_float_spect original_spec(
    to_floatbv_type(type).fraction, to_floatbv_type(type).exponent);
  v.value.change_spec(original_spec);

  assert(!v.value.is_NaN() && !v.value.is_infinity());
  if (upper)
    v.value.increment(true);
  else
    v.value.decrement(true);

  return value;
}

template <>
expr2tc interval_domaint::make_expression_value<wrapped_interval>(
  const wrapped_interval &interval,
  const type2tc &type,
  bool upper) const
{
  return from_integer(
    upper ? interval.get_upper() : interval.get_lower(), type);
}

template <>
expr2tc interval_domaint::make_expression_helper<wrapped_interval>(
  const expr2tc &symbol) const
{
  symbol2t src = to_symbol2t(symbol);

  if (!is_mapped<wrapped_interval>(src))
    return gen_true_expr();
  const auto interval = get_interval_from_symbol<wrapped_interval>(src);

  if (interval.is_top())
    return gen_true_expr();

  if (interval.is_bottom())
    return gen_false_expr();

  std::vector<expr2tc> conjuncts;
  assert(src.type == interval.t);

  if (interval.singleton())
  {
    expr2tc value = make_expression_value(interval, src.type, true);
    conjuncts.push_back(equality2tc(symbol, value));
  }
  else
  {
    assert(interval.upper && interval.lower);
    // Interval: [a,b]
    std::vector<expr2tc> disjuncts;

    auto convert = [this, &src, &symbol, &disjuncts](wrapped_interval &w) {
      assert(*w.lower <= *w.upper);

      std::vector<expr2tc> s_conjuncts;
      expr2tc value1 = make_expression_value(w, src.type, true);
      if (w.singleton())
      {
        disjuncts.push_back(equality2tc(symbol, value1));
        return;
      }
      s_conjuncts.push_back(lessthanequal2tc(symbol, value1));
      expr2tc value2 = make_expression_value(w, src.type, false);
      s_conjuncts.push_back(lessthanequal2tc(value2, symbol));
      disjuncts.push_back(conjunction(s_conjuncts));
    };

    for (auto &c : wrapped_interval::cut(interval))
    {
      convert(c);
    }
    conjuncts.push_back(disjunction(disjuncts));
  }
  return conjunction(conjuncts);
}

template <class T>
expr2tc interval_domaint::make_expression_helper(const expr2tc &symbol) const
{
  symbol2t src = to_symbol2t(symbol);

  if (!is_mapped<T>(src))
    return gen_true_expr();
  T interval = get_interval_from_symbol<T>(src);

  // Some intervals can be beyond what the type can hold
  // e.g., [-infinity,256] for an unsigned char.
  // Althugh this is expected (when modular intervals are off)
  // We still need to deal with the out-of-bounds when generating
  // the expressions, as 256 would be converted to 0.
  T type_interval = generate_modular_interval<T>(src);
  interval.intersect_with(type_interval);

  if (interval.is_top())
    return gen_true_expr();

  if (interval.is_bottom())
    return gen_false_expr();

  std::vector<expr2tc> conjuncts;
  auto typecast = [&symbol](expr2tc v) {
    c_implicit_typecast(v, symbol->type, *migrate_namespace_lookup);
    return v;
  };
  if (interval.singleton())
  {
    expr2tc value = make_expression_value(interval, src.type, true);
    conjuncts.push_back(equality2tc(typecast(value), symbol));
  }
  else
  {
    if (interval.upper)
    {
      expr2tc value = make_expression_value(interval, src.type, true);
      conjuncts.push_back(lessthanequal2tc(symbol, typecast(value)));
    }

    if (interval.lower)
    {
      expr2tc value = make_expression_value(interval, src.type, false);
      conjuncts.push_back(lessthanequal2tc(typecast(value), symbol));
    }
  }
  return conjunction(conjuncts);
}

// END TEMPLATES

// TODO: refactor
void interval_domaint::output(std::ostream &out) const
{
  if (bottom)
  {
    out << "BOTTOM\n";
    return;
  }

  for (const auto &interval : int_map)
  {
    if (interval.second.is_top())
      continue;
    if (interval.second.lower)
      out << *interval.second.lower << " <= ";
    out << interval.first;
    if (interval.second.upper)
      out << " <= " << *interval.second.upper;
    out << "\n";
  }

  for (const auto &interval : wrap_map)
  {
    out << interval.second.get_lower() << " <= ";
    out << interval.first;

    out << " <= " << interval.second.get_upper();
    out << "\n";
  }

  for (const auto &interval : real_map)
  {
    if (interval.second.is_top())
      continue;
    if (interval.second.lower)
      out << *interval.second.lower << " <= ";
    out << interval.first;
    if (interval.second.upper)
      out << " <= " << *interval.second.upper;
    out << "\n";
  }
}

bool contains_float(const expr2tc &e)
{
  if (is_floatbv_type(e->type))
    return true;

  bool inner_float = false;
  e->foreach_operand([&inner_float](auto &it) {
    if (contains_float(it))
      inner_float = true;
  });

  return inner_float;
}

void interval_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &ns)
{
  (void)ns;

  const goto_programt::instructiont &instruction = *from;
  switch (instruction.type)
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
    if (from->targets.front() != next) // If equal then a skip
    {
      if (next == to)
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

  case RETURN:
  {
    // After a return, all function arguments becomes nondet
    const symbolt *current_function = ns.lookup(instruction.function);
    type2tc t = migrate_type(current_function->type);
    const code_type2t &function = to_code_type(t);

    for (size_t i = 0; i < function.arguments.size(); i++)
    {
      const type2tc &arg_type = function.arguments[i];
      const expr2tc arg_symbol =
        symbol2tc(arg_type, function.argument_names[i]);
      havoc_rec(arg_symbol);
    }

    /* The current implementation of the abstract interpreter do not store
     * the return variable (which would be too tricky anyway). We can deal
     * with this by constructing a tmp symbol in which we can apply assumptions
     * later */
    expr2tc return_var = symbol2tc(
      function.ret_type, fmt::format("c:{}:ret", instruction.function));

    assign(
      code_assign2tc(return_var, to_code_return2t(instruction.code).operand));
    break;
  }

  case ASSERT:
  {
    // There is a bug in Floats that need to be investigated! regression-float/nextafter
    if (!contains_float(instruction.guard) && enable_assume_asserts)
      assume(instruction.guard);
    break;
  }

  case FUNCTION_CALL:
  case END_FUNCTION:
  case ATOMIC_BEGIN:
  case ATOMIC_END:
  case NO_INSTRUCTION_TYPE:
  case OTHER:
  case SKIP:
  case LOCATION:
  case THROW: // TODO: try/catch intervals
  case CATCH: // TODO: try/catch intervals
  case DEAD:
  case THROW_DECL:
  case THROW_DECL_END:
    break;
  }

  /* The abstract interpreter can only affect the state 'after' the execution of the statement
   * however, function calls need to change the parameter 'before' its execution. We can
   * deal with this by just checking if the target instruction is a function call!
   */
  if (to->is_function_call())
  {
    const code_function_call2t &code_function_call =
      to_code_function_call2t(to->code);

    // We don't know anything about the return value
    if (!is_nil_expr(code_function_call.ret))
    {
      havoc_rec(code_function_call.ret);
    }

    assert(is_code_type(code_function_call.function->type));
    const code_type2t &function =
      to_code_type(code_function_call.function->type);

    // Let's do an assignment for all parameters!
    for (size_t i = 0; i < function.arguments.size(); i++)
    {
      const expr2tc &arg_value = code_function_call.operands[i];
      const type2tc &arg_type = function.arguments[i];
      const expr2tc arg_symbol =
        symbol2tc(arg_type, function.argument_names[i]);

      // Are we dealing with a recursive function?
      std::unordered_set<expr2tc, irep2_hash> symbols;
      get_symbols(arg_value, symbols);

      bool is_recursive_arg = symbols.count(arg_symbol);
      assign(code_assign2tc(arg_symbol, arg_value), is_recursive_arg);
    }
  }

  // Let's deal with returns now.
  to--;
  if (from->is_end_function() && to->is_function_call())
  {
    // TODO: deal with recursive functions
    if (from->function == to->function)
      return;

    const code_function_call2t &code_function_call =
      to_code_function_call2t(to->code);

    // Apply assignment over return value
    if (!is_nil_expr(code_function_call.ret))
    {
      expr2tc return_var = symbol2tc(
        code_function_call.ret->type,
        fmt::format("c:{}:ret", instruction.function));
      assign(code_assign2tc(code_function_call.ret, return_var));
    }
  }
}

template <class IntervalMap>
bool interval_domaint::join(
  IntervalMap &new_map,
  const IntervalMap &previous_map)
{
  bool result = false;
  for (auto new_it = new_map.begin(); new_it != new_map.end();) // no new_it++
  {
    // search for the variable that needs to be merged
    // containers have different sizes and ordering
    const auto b_it = previous_map.find(new_it->first);
    const auto f_it = fixpoint_map.find(new_it->first);
    if (b_it == previous_map.end())
    {
      new_it = new_map.erase(new_it);
      if (f_it != fixpoint_map.end())
        fixpoint_map.erase(f_it);
      result = true;
    }
    else
    {
      auto previous = new_it->second; // [0,0] ... [0, +inf]
      auto after = b_it->second;      // [1,100] ... [1, 100]
      new_it->second.join(after);     // HULL // [0,100] ... [0, +inf]
      // Did we reach a fixpoint?
      if (new_it->second != previous)
      {
        if (f_it != fixpoint_map.end())
          f_it->second += 1;
        else
          fixpoint_map[new_it->first] = 0;

        result = true;
        // Try to extrapolate
        if (
          widening_extrapolate && fixpoint_map[new_it->first] > fixpoint_limit)
        {
          new_it->second = extrapolate_intervals(
            previous,
            new_it
              ->second); // ([0,0], [0,100] -> [0,inf]) ... ([0,inf], [0,100] --> [0,inf])
        }
      }

      else
      {
        // Found a fixpoint, we might try to narrow now!
        if (widening_narrowing)
        {
          after = interpolate_intervals(
            new_it->second,
            b_it
              ->second); // ([0,100], [1,100] --> [0,100] ... ([0,inf], [1,100] --> [0,100]))
          result |= new_it->second != after;
          new_it->second = after;
        }
      }

      new_it++;
    }
  }
  return result;
}

bool interval_domaint::join(const interval_domaint &b)
{
  if (b.is_bottom())
    return false;
  if (is_bottom())
  {
    *this = b;
    return true;
  }

  bool result = join(int_map, b.int_map) || join(real_map, b.real_map) ||
                join(wrap_map, b.wrap_map);
  return result;
}

void interval_domaint::assign(const expr2tc &expr, const bool recursive)
{
  assert(is_code_assign2t(expr));
  auto const &c = to_code_assign2t(expr);
  auto isbvop = is_bv_type(c.source) && is_bv_type(c.target);
  auto isfloatbvop = is_floatbv_type(c.source) && is_floatbv_type(c.target);

  if (!is_symbol2t(c.target))
  {
    if (is_dereference2t(c.target))
      clear_state();
    return;
  }

  if (isbvop)
  {
    if (enable_wrapped_intervals)
      apply_assignment<wrapped_interval>(c.target, c.source, recursive);
    else
      apply_assignment<integer_intervalt>(c.target, c.source, recursive);
  }
  else if (isfloatbvop && enable_real_intervals)
    apply_assignment<real_intervalt>(c.target, c.source, recursive);
}

void interval_domaint::havoc_rec(const expr2tc &expr)
{
  if (is_if2t(expr))
  {
    havoc_rec(to_if2t(expr).true_value);
    havoc_rec(to_if2t(expr).false_value);
  }
  else if (is_typecast2t(expr))
  {
    havoc_rec(to_typecast2t(expr).from);
  }
  else if (is_symbol2t(expr) || is_code_decl2t(expr))
  {
    // Reset the interval domain if it is being reasigned (-infinity, +infinity).
    irep_idt identifier = is_symbol2t(expr) ? to_symbol2t(expr).thename
                                            : to_code_decl2t(expr).value;
    if (is_bv_type(expr))
    {
      if (enable_wrapped_intervals)
        wrap_map.erase(identifier);
      else
        int_map.erase(identifier);
    }
    if (is_floatbv_type(expr))
      real_map.erase(identifier);
  }
  else
    log_debug("interval", "[havoc_rec] Missing support: {}", *expr);
}

void interval_domaint::assume_rec(
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
  {
    if (enable_wrapped_intervals)
      apply_assume_less<wrapped_interval>(lhs, rhs);
    else
      apply_assume_less<integer_intervalt>(lhs, rhs);
  }
  else if (
    is_floatbv_type(lhs) && is_floatbv_type(rhs) && enable_real_intervals)
    apply_assume_less<real_intervalt>(lhs, rhs);
}

void interval_domaint::assume(const expr2tc &cond)
{
  expr2tc new_cond = cond;
  simplify(new_cond);

#if 0
  // Let's check whether this condition is always false
  if(
    enable_eval_assumptions &&
    eval_boolean_expression(new_cond, *this).is_false())
  {
    log_debug("interval", "The expr {} is always false. Returning bottom", *cond);
    make_bottom();
    return;
  }
#endif

#ifdef ENABLE_GOTO_CONTRACTOR
  /// use ibex contractors to reduce the intervals for interval analysis
  if (enable_ibex_contractor)
  {
    interval_analysis_ibex_contractor contractor;
    if (contractor.parse_guard(new_cond))
    {
      contractor.maps_to_domains(int_map, real_map);
      contractor.apply_contractor();
      new_cond = contractor.result_of_outer();
      simplify(new_cond);
    }
  }
#endif

  assume_rec(new_cond, false);
}

tvt interval_domaint::eval_boolean_expression(
  const expr2tc &cond,
  const interval_domaint &id)
{
  // TODO: for now we will only support integer expressions (no mix!)
  if (enable_wrapped_intervals)
  {
    log_debug("interval", "[eval_boolean_expression] Disabled for wrapped");
    return tvt(tvt::TV_UNKNOWN);
  }

  if (contains_float(cond))
  {
    log_debug(
      "interval", "[eval_boolean_expression] No support for floats/mixing");
    return tvt(tvt::TV_UNKNOWN);
  }

  auto interval = id.get_interval<integer_intervalt>(cond);

  // If the interval does not contain zero then it's always true
  if (!interval.contains(0))
    return tvt(tvt::TV_TRUE);

  // If it does contain zero and its singleton then it's always false
  if (interval.singleton())
    return tvt(tvt::TV_FALSE);

  return tvt(tvt::TV_UNKNOWN);
}

void interval_domaint::assume_rec(const expr2tc &cond, bool negation)
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
    if (is_bv_type(cond) || is_bool_type(cond))
    {
      if (enable_wrapped_intervals)
        apply_assume_symbol_truth<wrapped_interval>(
          to_symbol2t(cond), negation);
      else
        apply_assume_symbol_truth<integer_intervalt>(
          to_symbol2t(cond), negation);
    }
    else if (is_floatbv_type(cond) && enable_real_intervals)
      apply_assume_symbol_truth<real_intervalt>(to_symbol2t(cond), negation);
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

void interval_domaint::dump() const
{
  std::ostringstream oss;
  output(oss);
  log_status("{}", oss.str());
}
expr2tc interval_domaint::make_expression(const expr2tc &symbol) const
{
  assert(is_symbol2t(symbol));
  if (is_bv_type(symbol))
  {
    if (enable_wrapped_intervals)
      return make_expression_helper<wrapped_interval>(symbol);
    else
      return make_expression_helper<integer_intervalt>(symbol);
  }
  if (is_floatbv_type(symbol) && enable_real_intervals)
    return make_expression_helper<real_intervalt>(symbol);
  return gen_true_expr();
}

// TODO: simplify
bool interval_domaint::ai_simplify(expr2tc &condition, const namespacet &ns)
  const
{
  (void)ns;

  if (!enable_assertion_simplification)
    return true;

  tvt eval = eval_boolean_expression(condition, *this);
  if (eval.is_true())
  {
    // TODO: convert to 1?
  }

  if (eval.is_false())
  {
    // TODO: convert to 0?
  }

  // TODO: contract expression (implication?)

  return false;
}

void interval_domaint::set_options(const optionst &options)
{
  enable_interval_arithmetic =
    options.get_bool_option("interval-analysis-arithmetic");
  enable_interval_bitwise_arithmetic =
    options.get_bool_option("interval-analysis-bitwise");
  enable_modular_intervals =
    options.get_bool_option("interval-analysis-modular");
  enable_assertion_simplification =
    options.get_bool_option("interval-analysis-simplify");
  enable_contraction_for_abstract_states =
    !options.get_bool_option("interval-analysis-no-contract");
  enable_wrapped_intervals =
    options.get_bool_option("interval-analysis-wrapped");
  enable_assume_asserts =
    options.get_bool_option("interval-analysis-assume-asserts");
  enable_eval_assumptions =
    options.get_bool_option("interval-analysis-eval-assumptions");
  enable_ibex_contractor =
    options.get_bool_option("interval-analysis-ibex-contractor");

  auto fixpoint_str = options.get_option("interval-analysis-extrapolate-limit");
  fixpoint_limit = fixpoint_str.empty() ? 5 : atoi(fixpoint_str.c_str());

  widening_extrapolate =
    options.get_bool_option("interval-analysis-extrapolate");
  widening_under_approximate_bound =
    options.get_bool_option("interval-analysis-extrapolate-under-approximate");
  widening_narrowing = options.get_bool_option("interval-analysis-narrowing");
}

// Options
bool interval_domaint::enable_interval_arithmetic = false;
bool interval_domaint::enable_interval_bitwise_arithmetic = false;
bool interval_domaint::enable_modular_intervals = false;
bool interval_domaint::enable_assertion_simplification = false;
bool interval_domaint::enable_contraction_for_abstract_states = true;
bool interval_domaint::enable_wrapped_intervals = false;
bool interval_domaint::enable_real_intervals = true;
bool interval_domaint::enable_assume_asserts = true;
bool interval_domaint::enable_eval_assumptions = true;
bool interval_domaint::enable_ibex_contractor = false;

// Widening options
unsigned interval_domaint::fixpoint_limit = 5;
bool interval_domaint::widening_under_approximate_bound = false;
bool interval_domaint::widening_extrapolate = false;
bool interval_domaint::widening_narrowing = false;
