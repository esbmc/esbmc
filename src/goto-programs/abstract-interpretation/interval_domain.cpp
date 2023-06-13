/// \file
/// Interval Domain
// TODO: Ternary operators, lessthan into lessthanequal for integers
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/std_expr.h>

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
  //  if(sym.thename)
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

template <>
integer_intervalt
interval_domaint::get_interval_from_const(const expr2tc &e) const
{
  integer_intervalt result; // (-infinity, infinity)
  if(!is_constant_int2t(e))
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
  if(!is_constant_floatbv2t(e))
    return result;

  auto real_value = to_constant_floatbv2t(e).value;

  // Health check, is the convertion to double ok? See #1037
  if(!std::isnormal(real_value.to_double()) || real_value.is_zero())
  {
    if(real_value.is_double())
      log_warning("ESBMC fails to to convert {} into double", *e);

    // Give up for top!
    return result;
  }

  auto value1 = to_constant_floatbv2t(e).value;
  auto value2 = to_constant_floatbv2t(e).value;
  value1.increment(true);
  value2.decrement(true);

  if(value1.is_NaN() || value1.is_infinity())
  {
    assert(result.is_top() && !result.is_bottom());
    return result;
  }

  // [value2, value1]
  // a <= value1
  result.make_le_than(value1.to_double());
  // a >= value2
  result.make_ge_than(value2.to_double());

  assert(!result.is_bottom());
  return result;
}

template <>
wrapped_interval
interval_domaint::get_interval_from_const(const expr2tc &e) const
{
  wrapped_interval result(e->type); // [0, 2^(length(e->type)) - 1]
  if(!is_constant_int2t(e))
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
  BigInt b;
  integer_intervalt result;
  if(is_unsignedbv_type(t))
  {
    b.setPower2(t->get_width());
    result.make_le_than(b - 1);
    result.make_ge_than(0);
  }
  else if(is_signedbv_type(t))
  {
    b.setPower2(t->get_width() - 1);
    result.make_ge_than(-b);
    b = b - 1;
    result.make_le_than(b);
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
void interval_domaint::apply_assignment(const expr2tc &lhs, const expr2tc &rhs)
{
  assert(is_symbol2t(lhs));
  // a = b
  auto b = get_interval<T>(rhs);
  if(enable_modular_intervals)
  {
    auto a = generate_modular_interval<T>(to_symbol2t(lhs));
    b.intersect_with(a);
  }

  // TODO: add classic algorithm
  update_symbol_interval(to_symbol2t(lhs), b);
}

template <class T>
T interval_domaint::extrapolate_intervals(const T &before, const T &after)
{
  T result; // Full extrapolation

  bool lower_decreased =
    !after.lower_set ||
    (before.lower_set && after.lower_set && after.lower < before.lower);
  bool upper_increased =
    !after.upper_set ||
    (before.upper_set && after.upper_set && after.upper > before.upper);

  if((lower_decreased || upper_increased) && !widening_under_approximate_bound)
    return result;

  // Set lower bound: if we didn't decrease then just update the interval
  if(!lower_decreased)
  {
    result.lower_set = before.lower_set;
    result.lower = before.lower;
  }

  // Set upper bound:
  if(!upper_increased)
  {
    result.upper_set = before.upper_set;
    result.upper = before.upper;
  }

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

  bool lower_increased = !before.lower_set;
  bool upper_decreased = !before.upper_set;

  result.lower_set = lower_increased ? after.lower_set : before.lower_set;
  result.lower = lower_increased ? after.lower : before.lower;

  result.upper_set = upper_decreased ? after.upper_set : before.upper_set;
  result.upper = upper_decreased ? after.upper : before.upper;

  return result;
}

template <class T>
T interval_domaint::get_interval(const expr2tc &e) const
{
  // This needs to come before constant number
  if(is_constant_bool2t(e))
  {
    auto r = get_top_interval_from_expr<T>(e);
    r.lower = to_constant_bool2t(e).is_true();
    r.upper = to_constant_bool2t(e).is_true();
    return r;
  }

  if(is_symbol2t(e))
    return get_interval_from_symbol<T>(to_symbol2t(e));

  if(is_neg2t(e))
    return -get_interval<T>(to_neg2t(e).value);

  if(is_constant_number(e))
    return get_interval_from_const<T>(e);

  if(is_if2t(e))
  {
    auto cond = get_interval<T>(to_if2t(e).cond);
    auto lhs = get_interval<T>(to_if2t(e).true_value);
    auto rhs = get_interval<T>(to_if2t(e).false_value);
    return T::ternary_if(cond, lhs, rhs);
  }

  if(is_typecast2t(e))
  {
    auto inner = get_interval<T>(to_typecast2t(e).from);
    return T::cast(inner, to_typecast2t(e).type);
  }

  // Arithmetic?
  auto arith_op = std::dynamic_pointer_cast<arith_2ops>(e);
  if(arith_op && enable_interval_arithmetic)
  {
    auto lhs = get_interval<T>(arith_op->side_1);
    auto rhs = get_interval<T>(arith_op->side_2);

    if(enable_interval_arithmetic)
    {
      if(is_add2t(e))
        return lhs + rhs;

      if(is_sub2t(e))
        return lhs - rhs;

      if(is_mul2t(e))
        return lhs * rhs;

      if(is_div2t(e))
        return lhs / rhs;

      if(is_modulus2t(e))
        return lhs % rhs;
    }
  }

  if(enable_interval_bitwise_arithmetic)
  {
    if(is_shl2t(e))
    {
      auto k = get_interval<T>(to_shl2t(e).side_2);
      auto i = get_interval<T>(to_shl2t(e).side_1);
      return T::left_shift(i, k);
    }

    if(is_ashr2t(e))
    {
      auto k = get_interval<T>(to_ashr2t(e).side_2);
      auto i = get_interval<T>(to_ashr2t(e).side_1);
      return T::arithmetic_right_shift(i, k);
    }

    if(is_lshr2t(e))
    {
      auto k = get_interval<T>(to_lshr2t(e).side_2);
      auto i = get_interval<T>(to_lshr2t(e).side_1);
      return T::logical_right_shift(i, k);
    }

    if(is_bitor2t(e))
    {
      auto rhs = get_interval<T>(to_bitor2t(e).side_2);
      auto lhs = get_interval<T>(to_bitor2t(e).side_1);
      return lhs | rhs;
    }

    if(is_bitand2t(e))
    {
      auto rhs = get_interval<T>(to_bitand2t(e).side_2);
      auto lhs = get_interval<T>(to_bitand2t(e).side_1);
      return lhs & rhs;
    }
    if(is_bitxor2t(e))
    {
      auto rhs = get_interval<T>(to_bitxor2t(e).side_2);
      auto lhs = get_interval<T>(to_bitxor2t(e).side_1);
      return lhs ^ rhs;
    }

    if(is_bitnot2t(e))
    {
      auto lhs = get_interval<T>(to_bitnot2t(e).value);
      return T::bitnot(lhs);
    }
  }

  // We could not generate from the expr. Return top
  return get_top_interval_from_expr<T>(e);
}

template <class Interval>
void interval_domaint::apply_assume_less(const expr2tc &a, const expr2tc &b)
{
  // 1. Apply contractor algorithms
  // 2. Update refs
  auto lhs = get_interval<Interval>(a);
  auto rhs = get_interval<Interval>(b);

  // TODO: Add less than equal
  if(enable_contraction_for_abstract_states)
    Interval::contract_interval_le(lhs, rhs);
  else
  {
    if(is_symbol2t(a) && is_symbol2t(b))
      lhs.make_sound_le(rhs);
    else
    {
      if(rhs.upper_set)
        lhs.make_le_than(rhs.get_upper());

      if(lhs.lower_set)
        rhs.make_ge_than(lhs.get_lower());
    }
  }
  // No need for widening, this is a restriction!
  if(is_symbol2t(a))
    update_symbol_interval<Interval>(to_symbol2t(a), lhs);

  if(is_symbol2t(b))
    update_symbol_interval<Interval>(to_symbol2t(b), rhs);

  if(rhs.is_bottom() || lhs.is_bottom())
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
  if(is_symbol2t(a))
    update_symbol_interval(to_symbol2t(a), s);

  if(is_symbol2t(b))
    update_symbol_interval(to_symbol2t(b), t);

  if(s.is_bottom() || t.is_bottom())
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
  bool upper) const
{
  return from_integer(upper ? interval.upper : interval.lower, type);
}

template <>
expr2tc interval_domaint::make_expression_value<real_intervalt>(
  const real_intervalt &interval,
  const type2tc &type,
  bool upper) const
{
  constant_floatbv2tc value(ieee_floatt(ieee_float_spect(
    to_floatbv_type(type).fraction, to_floatbv_type(type).exponent)));

  const auto d = (upper ? interval.upper : interval.lower).convert_to<double>();
  value->value.from_double(d);
  assert(!value->value.is_NaN() && !value->value.is_infinity());
  if(upper)
    value->value.increment(true);
  else
    value->value.decrement(true);

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

  if(!is_mapped<wrapped_interval>(src))
    return gen_true_expr();
  const auto interval = get_interval_from_symbol<wrapped_interval>(src);

  if(interval.is_top())
    return gen_true_expr();

  if(interval.is_bottom())
    return gen_false_expr();

  std::vector<expr2tc> conjuncts;
  assert(src.type == interval.t);

  if(interval.singleton())
  {
    expr2tc value = make_expression_value(interval, src.type, true);
    conjuncts.push_back(equality2tc(symbol, value));
  }
  else
  {
    assert(interval.upper_set && interval.lower_set);
    // Interval: [a,b]
    std::vector<expr2tc> disjuncts;

    auto convert = [this, &src, &symbol, &disjuncts](wrapped_interval &w) {
      assert(w.lower <= w.upper);

      std::vector<expr2tc> s_conjuncts;
      expr2tc value1 = make_expression_value(w, src.type, true);
      if(w.singleton())
      {
        disjuncts.push_back(equality2tc(symbol, value1));
        return;
      }
      s_conjuncts.push_back(lessthanequal2tc(symbol, value1));
      expr2tc value2 = make_expression_value(w, src.type, false);
      s_conjuncts.push_back(lessthanequal2tc(value2, symbol));
      disjuncts.push_back(conjunction(s_conjuncts));
    };

    for(auto &c : wrapped_interval::cut(interval))
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

  if(!is_mapped<T>(src))
    return gen_true_expr();
  const auto interval = get_interval_from_symbol<T>(src);
  if(interval.is_top())
    return gen_true_expr();

  if(interval.is_bottom())
    return gen_false_expr();

  std::vector<expr2tc> conjuncts;
  auto typecast = [&symbol](expr2tc v) {
    c_implicit_typecast(v, symbol->type, *migrate_namespace_lookup);
    return v;
  };
  if(interval.singleton())
  {
    expr2tc value = make_expression_value(interval, src.type, true);
    conjuncts.push_back(equality2tc(typecast(value), symbol));
  }
  else
  {
    if(interval.upper_set)
    {
      expr2tc value = make_expression_value(interval, src.type, true);
      conjuncts.push_back(lessthanequal2tc(symbol, typecast(value)));
    }

    if(interval.lower_set)
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

  for(const auto &interval : wrap_map)
  {
    out << interval.second.get_lower() << " <= ";
    out << interval.first;

    out << " <= " << interval.second.get_upper();
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

template <class IntervalMap>
bool interval_domaint::join(
  IntervalMap &new_map,
  const IntervalMap &previous_map)
{
  bool result = false;
  for(auto new_it = new_map.begin(); new_it != new_map.end();) // no new_it++
  {
    // search for the variable that needs to be merged
    // containers have different sizes and ordering
    const auto b_it = previous_map.find(new_it->first);
    const auto f_it = fixpoint_map.find(new_it->first);
    if(b_it == previous_map.end())
    {
      new_it = new_map.erase(new_it);
      if(f_it != fixpoint_map.end())
        fixpoint_map.erase(f_it);
      result = true;
    }
    else
    {
      auto previous = new_it->second; // [0,0] ... [0, +inf]
      auto after = b_it->second;      // [1,100] ... [1, 100]
      new_it->second.join(after);     // HULL // [0,100] ... [0, +inf]
      // Did we reach a fixpoint?
      if(new_it->second != previous)
      {
        if(f_it != fixpoint_map.end())
          f_it->second += 1;
        else
          fixpoint_map[new_it->first] = 0;

        result = true;
        // Try to extrapolate
        if(widening_extrapolate && fixpoint_map[new_it->first] > fixpoint_limit)
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
        if(widening_narrowing)
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
  if(b.is_bottom())
    return false;
  if(is_bottom())
  {
    *this = b;
    return true;
  }

  bool result = join(int_map, b.int_map) || join(real_map, b.real_map) ||
                join(wrap_map, b.wrap_map);
  return result;
}

void interval_domaint::assign(const expr2tc &expr)
{
  assert(is_code_assign2t(expr));
  auto const &c = to_code_assign2t(expr);
  auto isbvop = is_bv_type(c.source) && is_bv_type(c.target);
  auto isfloatbvop = is_floatbv_type(c.source) && is_floatbv_type(c.target);

  if(!is_symbol2t(c.target))
    return;
  if(isbvop)
  {
    if(enable_wrapped_intervals)
      apply_assignment<wrapped_interval>(c.target, c.source);
    else
      apply_assignment<integer_intervalt>(c.target, c.source);
  }
  if(isfloatbvop && enable_real_intervals)
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
    if(is_bv_type(expr))
    {
      if(enable_wrapped_intervals)
        wrap_map.erase(identifier);
      else
        int_map.erase(identifier);
    }
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
  {
    if(enable_wrapped_intervals)
      apply_assume_less<wrapped_interval>(lhs, rhs);
    else
      apply_assume_less<integer_intervalt>(lhs, rhs);
  }
  else if(is_floatbv_type(lhs) && is_floatbv_type(rhs) && enable_real_intervals)
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
  log_status("{}", oss.str());
}
expr2tc interval_domaint::make_expression(const expr2tc &symbol) const
{
  assert(is_symbol2t(symbol));
  if(is_bv_type(symbol))
  {
    if(enable_wrapped_intervals)
      return make_expression_helper<wrapped_interval>(symbol);
    else
      return make_expression_helper<integer_intervalt>(symbol);
  }
  if(is_floatbv_type(symbol) && enable_real_intervals)
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

// Widening options
unsigned interval_domaint::fixpoint_limit = 5;
bool interval_domaint::widening_under_approximate_bound = false;
bool interval_domaint::widening_extrapolate = false;
bool interval_domaint::widening_narrowing = false;
