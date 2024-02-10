#pragma once

#include "interval_template.h"

class wrapped_interval : public interval_templatet<BigInt>
{
public:
  wrapped_interval() : upper_bound(0)
  {
  }

  explicit wrapped_interval(const type2tc &t)
    : t(t), upper_bound(compute_upper_bound(t))
  {
    assert(
      is_signedbv_type(t) || is_unsignedbv_type(t) || is_bool_type(t) ||
      is_pointer_type(t));

    lower = 0;
    upper = is_bool_type(t) ? 1 : compute_upper_bound(t) - 1;
  }

  BigInt convert_to_wrap(const BigInt &b) const
  {
    auto value = b;
    // TODO: I could use a better way here!
    if (b.is_negative() && !b.is_zero())
      value += get_upper_bound();

    assert(value >= 0);
    assert(value <= get_upper_bound());
    return value;
  }

  void set(bool is_upper, const BigInt &b) override
  {
    auto value = convert_to_wrap(b);

    if (is_upper)
      upper = value;
    else
      lower = value;
  }

  BigInt get(bool is_upper) const override
  {
    auto is_signed = is_signedbv_type(t);
    auto value = is_upper ? upper : lower;

    if (is_signed)
    {
      auto middle = get_upper_bound() / 2; // [128] 256
      if (*value >= middle)
        *value -= middle * 2;
    }

    return *value;
  }

  bool bottom = false;
  bool empty() const override
  {
    return bottom;
  }

  bool is_top() const override
  {
    return !is_bottom() && cardinality() == get_upper_bound();
  }

  static wrapped_interval complement(const wrapped_interval &w)
  {
    wrapped_interval result(w.t);

    if (w.is_top())
      result.bottom = true;

    if (!w.is_bottom())
    {
      auto mod = w.get_upper_bound();
      result.lower = (*w.upper + 1) % mod;
      result.upper = (*w.lower - 1) % mod;
      if (result.upper < 0)
        *result.upper += mod;
    }
    return result;
  }

  /// b <=_a c iff b -_w a <= c -_w a
  static bool wrapped_le(
    const BigInt &b,
    const BigInt &a,
    const BigInt &c,
    const type2tc &t)
  {
    auto mod = compute_upper_bound(t);
    // https://torstencurdt.com/tech/posts/modulo-of-negative-numbers/
    auto lhs = (b - a) % mod;
    if (lhs < 0)
      lhs = lhs + mod;
    auto rhs = (c - a) % mod;
    if (rhs < 0)
      rhs = rhs + mod;

    return lhs <= rhs;
  }

  void make_le_than(const BigInt &v) override // add upper bound
  {
    if (get_upper_bound() <= v)
      return;
    wrapped_interval value(t);
    value.lower = get_min();

    value.set_upper(v);

    *this = over_meet(*this, value);
  }

  void make_le_than(const wrapped_interval &rhs)
  {
    if (rhs.is_bottom())
    {
      bottom = true;
      return;
    }

    // RHS contains upper bound (MAX_INT) we are always less than that
    if (rhs.contains(get_max()))
      return;

    wrapped_interval value(t);
    value.lower = get_min();
    value.upper = rhs.upper;

    *this = over_meet(*this, value);
  }

  void make_ge_than(const BigInt &v) override // add upper bound
  {
    wrapped_interval value(t);
    value.set_lower(v);
    value.upper = get_max();
    *this = over_meet(*this, value);
  }

  void make_ge_than(const wrapped_interval &rhs) // add upper bound
  {
    if (rhs.is_bottom())
    {
      bottom = true;
      return;
    }

    // RHS contains lower bound (MIN_INT) we are always greater than that
    if (rhs.contains(get_min()))
      return;

    wrapped_interval value(t);
    value.lower = rhs.lower;
    value.upper = get_max();

    *this = over_meet(*this, value);
  }

  BigInt get_min() const
  {
    // 0^w or 10^(w-1)
    return is_signedbv_type(t) ? get_upper_bound() / 2 : 0;
  }

  BigInt get_max() const
  {
    // 1^w or 01^(w-1)
    return is_signedbv_type(t) ? get_upper_bound() / 2 - 1
                               : get_upper_bound() - 1;
  }

  BigInt cardinality() const
  {
    if (is_bottom())
      return 0;

    auto mod = get_upper_bound();
    assert(mod != 0);
    auto card = (((*upper - *lower) % mod) + 1);
    if (card < 0)
      card += mod;
    return card;
  }

  bool contains(const BigInt &e) const override
  {
    if (is_top())
      return true;

    return wrapped_le(e, *lower, *upper, t);
  }

  bool is_equal(const wrapped_interval &rhs) const
  {
    return t == rhs.t && lower == rhs.lower && upper == rhs.upper;
  }

  bool is_included(const wrapped_interval &rhs) const
  {
    if (is_bottom() || rhs.is_top() || is_equal(rhs))
      return true;
    if (is_top() || rhs.is_bottom())
      return false;

    return rhs.contains(*lower) && rhs.contains(*upper) &&
           (!contains(*rhs.lower) || !contains(*rhs.upper));
  }

  static wrapped_interval
  extend(const wrapped_interval &s, const wrapped_interval &t)
  {
    if (s.is_included(t))
      return t;
    if (t.is_included(s))
      return s;

    wrapped_interval result(s.t);
    if (complement(s).is_included(t))
    {
      return result;
    }

    result.lower = s.lower;
    result.upper = t.upper;
    return result;
  }

  static wrapped_interval
  gap(const wrapped_interval &s, const wrapped_interval &t)
  {
    wrapped_interval result(s.t);
    if (
      !t.is_bottom() && !s.is_bottom() && !t.contains(*s.upper) &&
      !s.contains(*t.lower))
    {
      result.lower = t.lower;
      result.upper = s.upper;
      return complement(result);
    }
    result.bottom = true;
    return result;
  }

  static wrapped_interval
  bigger(const wrapped_interval &s, const wrapped_interval &t)
  {
    return t.cardinality() > s.cardinality() ? t : s;
  }

  /// Over union
  static wrapped_interval over_join(std::vector<wrapped_interval> &r)
  {
    wrapped_interval bottom;
    bottom.bottom = true;

    if (r.empty())
      return bottom;
    bottom.t = r[0].t;
    auto f = bottom;
    auto g = bottom;

    // Sort by left bound
    std::sort(r.begin(), r.end(), [](auto a, auto b) {
      if (a.is_bottom())
        return true;
      if (b.is_bottom())
        return false;

      return *a.lower < *b.lower;
    });

    for (unsigned i = 0; i < r.size(); ++i)
    {
      if (
        r[i].is_top() || (wrapped_le(*(r[i].upper), 0, *(r[i].lower), r[i].t)))
      {
        f = extend(f, r[i]);
      }
    }

    for (unsigned i = 0; i < r.size(); ++i)
    {
      g = bigger(g, gap(f, r[i]));
      f = extend(f, r[i]);
    }
    auto rr = complement(bigger(g, complement(f)));

    return rr;
  }
  static wrapped_interval
  over_join(const wrapped_interval &s, const wrapped_interval &t)
  {
    assert(s.t == t.t);
#if 0

    if(s.is_included(t))
      return t;
    if(t.is_included(s))
      return s;

    const BigInt &a = s.lower;
    const BigInt &b = s.upper;
    const BigInt &c = t.lower;
    const BigInt &d = t.upper;

    wrapped_interval result(s.t);

    if(t.contains(a) && t.contains(b) && s.contains(c) && s.contains(d))
      return result;

    if(t.contains(b) && s.contains(c))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    if(s.contains(d) && t.contains(a))
    {
      result.lower = c;
      result.upper = b;
      return result;
    }

    wrapped_interval b_c(s.t);
    b_c.lower = b;
    b_c.upper = c;
    wrapped_interval d_a(s.t);
    d_a.lower = d;
    d_a.upper = a;

    if(
      (b_c.cardinality() < d_a.cardinality()) ||
      (b_c.cardinality() == d_a.cardinality() && a <= c))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    result.lower = c;
    result.upper = b;

    return result;
#else
    std::vector<wrapped_interval> r;
    r.push_back(s);
    r.push_back(t);
    auto result = over_join(r);
    return result;
#endif
  }

  void approx_union_with(const interval_templatet<BigInt> &i) override
  {
    wrapped_interval rhs(t);
    rhs.bottom = i.is_bottom();
    rhs.lower = i.lower;
    rhs.upper = i.upper;

    *this = over_join(rhs, *this);
  }

  // Under meet
  static wrapped_interval
  under_meet(const wrapped_interval &s, const wrapped_interval &t)
  {
    return complement(over_join(complement(s), complement(t)));
  }

  static std::vector<wrapped_interval>
  intersection(const wrapped_interval &s, const wrapped_interval &t)
  {
    assert(s.t == t.t);

    std::vector<wrapped_interval> result;

    if (s.is_bottom() || t.is_bottom())
      return result;

    if (s.is_equal(t) || s.is_top())
    {
      result.push_back(t);
      return result;
    }

    if (t.is_top())
    {
      result.push_back(s);
      return result;
    }

    const BigInt &a = *s.lower;
    const BigInt &b = *s.upper;
    const BigInt &c = *t.lower;
    const BigInt &d = *t.upper;

    if (t.contains(a) && t.contains(b) && s.contains(c) && s.contains(d))
    {
      wrapped_interval a_d(s.t);
      a_d.lower = a;
      a_d.upper = d;

      wrapped_interval c_b(s.t);
      c_b.lower = c;
      c_b.upper = b;

      result.push_back(a_d);
      result.push_back(c_b);
      return result;
    }

    if (t.contains(a) && t.contains(b))
    {
      result.push_back(s);
      return result;
    }

    if (s.contains(c) && s.contains(d))
    {
      result.push_back(t);
      return result;
    }

    if (t.contains(a) && s.contains(d) && !t.contains(b) && !s.contains(c))
    {
      wrapped_interval a_d(s.t);
      a_d.lower = a;
      a_d.upper = d;

      result.push_back(a_d);
      return result;
    }

    if (t.contains(b) && s.contains(c) && !t.contains(a) && !s.contains(d))
    {
      wrapped_interval c_b(s.t);
      c_b.lower = c;
      c_b.upper = b;

      result.push_back(c_b);
      return result;
    }

    return result;
  }

  /// Over meet
  static wrapped_interval
  over_meet(const wrapped_interval &s, const wrapped_interval &t)
  {
    // TODO: this will probably not be needed!
    assert(s.t == t.t);

    if (s.is_included(t))
      return s;
    if (t.is_included(s))
      return t;

    const BigInt &a = *s.lower;
    const BigInt &b = *s.upper;
    const BigInt &c = *t.lower;
    const BigInt &d = *t.upper;

    wrapped_interval result(s.t);

    if (!t.contains(a) && !t.contains(b) && !s.contains(c) && !s.contains(d))
    {
      result.bottom = true;
      return result;
    }

    if (s.contains(d) && t.contains(a))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    if (t.contains(b) && s.contains(c))
    {
      result.lower = c;
      result.upper = b;
      return result;
    }

    if (
      (s.cardinality() < t.cardinality()) ||
      (s.cardinality() == t.cardinality() && a <= c))
    {
      return s;
    }

    result.lower = c;
    result.upper = d;
    return result;
  }

  friend wrapped_interval operator-(const wrapped_interval &lhs)
  {
    wrapped_interval result(lhs.t);
    result.set_lower(-1);
    result.set_upper(-1);
    return result * lhs;
  }

  static wrapped_interval
  extrapolate_to(const wrapped_interval &f1, const wrapped_interval &f2)
  {
    assert(f1.t == f2.t);
    wrapped_interval result(f2.t);
#if 1
    if (f2.is_included(f1))
      return f1;

    // The interval is too big already, give up to TOP
    if (f1.cardinality() >= f1.get_upper_bound() / 2)
      return result;

    auto f1_to_f2 = over_join(f1, f2);
    assert(!f1_to_f2.is_bottom());

    // Lower bound keeps the same but upper increases
    if (*f1_to_f2.lower == *f1.lower && *f1_to_f2.upper == *f2.upper)
    {
      wrapped_interval double_upper(f1.t);
      double_upper.lower = *f1_to_f2.lower;
      double_upper.upper =
        ((((2 * *f1.upper) - *f1.lower) % f1.get_upper_bound()) + 1) %
        f1.get_upper_bound();
      return over_join(f1_to_f2, double_upper);
    }

    // Upper bound keeps the same but lower decreases
    if (*f1_to_f2.lower == *f2.lower && *f1_to_f2.upper == *f1.upper)
    {
      wrapped_interval double_lower(f1.t);
      double_lower.lower =
        ((((2 * *f1.lower) - *f1.upper) % f1.get_upper_bound()) - 1) %
        f1.get_upper_bound();
      double_lower.upper = *f1.upper;
      return over_join(f1_to_f2, double_lower);
    }

    // Lower and upper bound of f1 is included in f2
    if (f2.contains(*f1.lower) && f2.contains(*f1.upper))
    {
      wrapped_interval magic(f1.t);
      // Maintain the lower
      magic.lower = *f2.lower;
      // Increase the upper by the difference between uppers
      magic.upper = (((*f2.upper + (((2 * *f1.upper) - (2 * *f1.lower)) %
                                    f1.get_upper_bound())) %
                      f1.get_upper_bound()) +
                     1) %
                    f1.get_upper_bound();
      return over_join(f1_to_f2, magic);
    }
#endif

    // Give up return top
    return result;
  }

  friend wrapped_interval
  operator+(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    assert(lhs.t->get_width() == rhs.t->get_width());
    wrapped_interval result(lhs.t);

    if (lhs.is_bottom() || rhs.is_bottom())
    {
      result.bottom = true;
      return result;
    }

    auto mod = lhs.get_upper_bound();
    if (lhs.cardinality() + rhs.cardinality() <= mod)
    {
      result.lower = (*lhs.lower + *rhs.lower) % mod;
      result.upper = (*lhs.upper + *rhs.upper) % mod;
      assert(*result.lower >= 0);
      assert(*result.upper >= 0);
    }

    return result;
  }

  friend wrapped_interval
  operator-(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    assert(lhs.t->get_width() == rhs.t->get_width());
    wrapped_interval result(lhs.t);

    if (lhs.is_bottom() || rhs.is_bottom())
    {
      result.bottom = true;
      return result;
    }

    auto mod = lhs.get_upper_bound();
    if (lhs.cardinality() + rhs.cardinality() <= mod)
    {
      result.lower = (*lhs.lower - *rhs.upper) % mod;
      result.upper = (*lhs.upper - *rhs.lower) % mod;
      if (*result.lower < 0)
        *result.lower += mod;
      if (*result.upper < 0)
        *result.upper += mod;
      assert(*result.lower >= 0);
      assert(*result.upper >= 0);
    }

    return result;
  }

  static wrapped_interval north_pole(const type2tc &t)
  {
    wrapped_interval result(t);

    result.lower = result.get_upper_bound() / 2 - 1;
    result.upper = result.get_upper_bound() / 2;

    return result;
  }

  static wrapped_interval south_pole(const type2tc &t)
  {
    wrapped_interval result(t);

    result.upper = 0;
    result.lower = result.get_upper_bound() - 1;

    return result;
  }

  std::vector<wrapped_interval> ssplit() const
  {
    std::vector<wrapped_interval> r;
    if (is_bottom())
      return r;

    if (is_top())
    {
      wrapped_interval north(t);
      wrapped_interval south(t);

      north.lower = get_upper_bound() / 2;
      north.upper = get_upper_bound() - 1;

      south.lower = 0;
      south.upper = get_upper_bound() / 2 - 1;

      r.push_back(north);
      r.push_back(south);
      return r;
    }

    if (!south_pole(t).is_included(*this))
    {
      r.push_back(*this);
      return r;
    }

    wrapped_interval north(t);
    wrapped_interval south(t);

    north.upper = upper;
    north.lower = 0;

    south.lower = lower;
    south.upper = get_upper_bound() - 1;

    r.push_back(north);
    r.push_back(south);

    return r;
  }

  static wrapped_interval
  difference(const wrapped_interval &s, const wrapped_interval &t)
  {
    return over_meet(s, complement(t));
  }

  std::vector<wrapped_interval> nsplit() const
  {
    std::vector<wrapped_interval> r;
    if (is_bottom())
      return r;

    if (is_top())
    {
      wrapped_interval north(t);
      wrapped_interval south(t);

      north.lower = get_upper_bound() / 2;
      north.upper = get_upper_bound() - 1;

      south.lower = 0;
      south.upper = get_upper_bound() / 2 - 1;

      r.push_back(north);
      r.push_back(south);
      return r;
    }

    if (!north_pole(t).is_included(*this))
    {
      r.push_back(*this);
      return r;
    }

    wrapped_interval north(t);
    wrapped_interval south(t);

    north.upper = upper;
    north.lower = get_upper_bound() / 2;

    south.lower = lower;
    south.upper = get_upper_bound() / 2 - 1;

    r.push_back(north);
    r.push_back(south);

    return r;
  }

  static std::vector<wrapped_interval> cut(const wrapped_interval &u)
  {
    std::vector<wrapped_interval> r;

    for (const auto &w : u.nsplit())
    {
      for (const auto &s : w.ssplit())
      {
        r.push_back(s);
      }
    }
    return r;
  }

  bool most_significant_bit(const BigInt &b) const
  {
    return (b >> (t->get_width() - 1)) == 1;
  }

  static BigInt trunc(const BigInt &b, unsigned k)
  {
    // TODO: and operation for bigint
    return b % BigInt::power2(k);
  }

  static wrapped_interval
  cast(const wrapped_interval &old, const type2tc &new_type)
  {
    // Special case for Bool!
    if (is_bool_type(new_type))
    {
      wrapped_interval boolean(new_type);
      boolean.lower = 0;
      boolean.upper = 1;

      if (!old.contains(0))
        boolean.lower = 1;

      else if (old.singleton())
        boolean.upper = 0;

      return boolean;
    }

    if (new_type->get_width() < old.t->get_width())
      return old.trunc(new_type);

    if (is_signedbv_type(old.t))
      return old.sign_extension(new_type);

    return old.zero_extension(new_type);
  }

  wrapped_interval zero_extension(const type2tc &cast) const
  {
    std::vector<wrapped_interval> parts;

    for (auto &interval : ssplit())
    {
      wrapped_interval result(cast);
      result.lower = interval.lower;
      result.upper = interval.upper;
      parts.push_back(result);
    }

    return over_join(parts);
  }

  wrapped_interval left_shift(unsigned k) const
  {
    if (is_bottom())
      return *this;

    wrapped_interval result(t);
    if (
      trunc(*lower, t->get_width() - k) == *lower &&
      trunc(*upper, t->get_width() - k) == *upper)
    {
      result.lower = *lower << k;
      result.upper = *upper << k;
    }
    else
    {
      result.lower = 0;
      result.upper = (get_upper_bound() - 1) - BigInt::power2m1(k);
    }
    return result;
  }

  static wrapped_interval
  left_shift(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    if (lhs.is_bottom())
      return lhs;

    if (rhs.lower == rhs.upper && !rhs.is_bottom())
      return lhs.left_shift(rhs.get_lower().to_uint64());

    wrapped_interval result(lhs.t);
    return result;
  }

  wrapped_interval logical_right_shift(unsigned k) const
  {
    if (is_bottom())
      return *this;

    wrapped_interval result(t);
    if (south_pole(t).is_included(*this))
    {
      result.lower = 0;
      result.upper = BigInt::power2m1(t->get_width() - k);
    }
    else
    {
      result.lower = *lower >> k;
      result.upper = *upper >> k;
    }
    return result;
  }

  static wrapped_interval
  logical_right_shift(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    if (lhs.is_bottom())
      return lhs;

    if (rhs.lower == rhs.upper && !rhs.is_bottom())
      return lhs.logical_right_shift(rhs.get_lower().to_uint64());

    wrapped_interval result(lhs.t);
    return result;
  }

  wrapped_interval arithmetic_right_shift(unsigned k) const
  {
    if (is_bottom())
      return *this;

    wrapped_interval result(t);
    if (north_pole(t).is_included(*this))
    {
      BigInt m = BigInt::power2m1(t->get_width() - k);
      result.lower = (get_upper_bound() - 1) - m;
      result.upper = m;
    }
    else
    {
      result.set_lower(get_lower() >> k);
      result.set_upper(get_upper() >> k);
    }
    return result;
  }

  static wrapped_interval arithmetic_right_shift(
    const wrapped_interval &lhs,
    const wrapped_interval &rhs)
  {
    if (lhs.is_bottom())
      return lhs;

    if (rhs.lower == rhs.upper && !rhs.is_bottom())
      return lhs.arithmetic_right_shift(rhs.get_lower().to_uint64());

    wrapped_interval result(lhs.t);
    return result;
  }

  static wrapped_interval
  equality(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    wrapped_interval result(lhs.t);
    if (
      lhs.singleton() && rhs.singleton() && lhs.get_lower() == rhs.get_lower())
    {
      result.set_lower(1);
      result.set_upper(1);
    }
    else if (intersection(lhs, rhs).empty())
    {
      result.set_lower(0);
      result.set_upper(0);
    }
    else
    {
      result.set_lower(0);
      result.set_upper(1);
    }
    return result;
  }

  static wrapped_interval invert_bool(const wrapped_interval &i)
  {
    wrapped_interval result(i.t);
    if (!i.contains(0))
    {
      // i is always true, return false
      result.set_lower(0);
      result.set_upper(0);
    }
    else if (i.singleton())
    {
      // i is always false, return true
      result.set_lower(1);
      result.set_upper(1);
    }
    else
    {
      // i is a maybe
      result.set_lower(0);
      result.set_upper(1);
    }
    return result;
  }

  static wrapped_interval
  not_equal(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    return invert_bool(equality(lhs, rhs));
  }

  /** @brief flatten all intervals cuts into a BigInt pair. 

      Examples (signed char):

      - get_interval_bounds([10, 127]) --> <10, 127>
      - get_interval_bounds([10, 128]) --> <-128, 10>
      - get_interval_bounds([10, 255]) --> <-128, 10>
      - get_interval_bounds([129, 130]) --> <-127, -126>
      - get_interval_bounds([255, 10]) --> <-1, 10>

      @warning This is not to represent a range interval, wrapped can have holes!
      From the example: [10, 128] contains both <-128, 10> but it does not contain 9 or -127!
  */
  std::pair<BigInt, BigInt> get_interval_bounds() const
  {
    BigInt minimum = get_lower();
    BigInt maximum = get_lower();

    for (const wrapped_interval &w : cut(*this))
    {
      BigInt local_min = std::min(w.get_lower(), w.get_upper());
      BigInt local_max = std::max(w.get_lower(), w.get_upper());

      minimum = std::min(minimum, local_min);
      maximum = std::max(maximum, local_max);
    }   
    return std::make_pair(minimum, maximum);
  }

  static wrapped_interval
  less_than(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    wrapped_interval result(lhs.t);
    const std::pair<BigInt, BigInt> lhs_bounds = lhs.get_interval_bounds();
    const std::pair<BigInt, BigInt> rhs_bounds = rhs.get_interval_bounds();

    result.set_lower(lhs_bounds.second < rhs_bounds.first ? 1 : 0);
    result.set_upper(lhs_bounds.first >= rhs_bounds.second ? 0 : 1);
    assert(result.get_lower() <= result.get_upper());
    return result;
  }

  static wrapped_interval
  less_than_equal(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    wrapped_interval result(lhs.t);
    const std::pair<BigInt, BigInt> lhs_bounds = lhs.get_interval_bounds();
    const std::pair<BigInt, BigInt> rhs_bounds = rhs.get_interval_bounds();

    result.set_lower(lhs_bounds.second <= rhs_bounds.first ? 1 : 0);
    result.set_upper(lhs_bounds.first > rhs_bounds.second ? 0 : 1);
    assert(result.get_lower() <= result.get_upper());
    return result;
  }

  static wrapped_interval
  greater_than_equal(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    return less_than(rhs, lhs);
  }

  static wrapped_interval
  greater_than(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    return less_than_equal(rhs, lhs);
  }

  wrapped_interval sign_extension(const type2tc &cast) const
  {
    std::vector<wrapped_interval> parts;

    auto compute_outer_region = [this, &cast](bool bit) {
      BigInt result = 0;
      if (bit)
        result = (compute_upper_bound(cast) - 1) - (get_upper_bound() - 1);

      return result;
    };

    for (auto &interval : nsplit())
    {
      wrapped_interval result(cast);
      result.lower =
        compute_outer_region(most_significant_bit(*interval.lower)) +
        *interval.lower;
      result.upper =
        compute_outer_region(most_significant_bit(*interval.upper)) +
        *interval.upper;
      parts.push_back(result);
    }

    return over_join(parts);
  }

  wrapped_interval trunc(const type2tc &cast) const
  {
    wrapped_interval result(cast);
    auto k = cast->get_width();
    assert(k <= t->get_width());
    if (is_bottom())
      result.bottom = true;

    else if (
      (*lower >> k) == (*upper >> k) && (trunc(*lower, k) <= trunc(*upper, k)))
    {
      result.lower = trunc(*lower, k);
      result.upper = trunc(*upper, k);
    }

    else if (
      ((*lower >> k) + 1 % 2) == (*upper >> k) % 2 &&
      (trunc(*lower, k) > trunc(*upper, k)))
    {
      result.lower = trunc(*lower, k);
      result.upper = trunc(*upper, k);
    }
    return result;
  }

  static std::vector<wrapped_interval>
  multiply_us(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    wrapped_interval result(lhs.t);

    wrapped_interval w_u(lhs.t);
    wrapped_interval w_s(lhs.t);

    auto up = lhs.get_upper_bound();

    auto &a = *lhs.lower;
    auto &b = *lhs.upper;
    auto &c = *rhs.lower;
    auto &d = *rhs.upper;

    if (b * d - a * c < up)
    {
      w_u.lower = (a * c) % up;
      w_u.upper = (b * d) % up;
    }

    if (
      ((w_s.most_significant_bit(a) == w_s.most_significant_bit(b)) ==
       (w_s.most_significant_bit(c) == w_s.most_significant_bit(d))) &&
      (b * d - a * c < up))
    {
      w_s.lower = a * c % up;
      w_s.upper = b * d % up;
    }
    else if (
      (w_s.most_significant_bit(a) && w_s.most_significant_bit(b)) &&
      (!w_s.most_significant_bit(c) && !w_s.most_significant_bit(d)) &&
      (b * c - a * d < up))
    {
      w_s.lower = a * d % up;
      w_s.upper = b * c % up;
    }
    else if (
      (!w_s.most_significant_bit(a) && !w_s.most_significant_bit(b)) &&
      (w_s.most_significant_bit(c) && w_s.most_significant_bit(d)) &&
      (a * d - b * c < up))
    {
      w_s.lower = b * c % up;
      w_s.upper = a * d % up;
    }

    return intersection(w_u, w_s);
  }

  friend wrapped_interval
  operator*(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    // TODO: over-join for list
    std::vector<wrapped_interval> r;
    for (auto &u : cut(lhs))
      for (auto &v : cut(rhs))
        for (auto &m : multiply_us(u, v))
          r.push_back(m);
    return over_join(r);
  }

  static wrapped_interval amb(const wrapped_interval &rhs)
  {
    wrapped_interval r(rhs.t);
    r.lower = 0;
    r.upper = *rhs.upper - 1;
    return r;
  }

  friend wrapped_interval
  operator%(const wrapped_interval &s, const wrapped_interval &t)
  {
    std::vector<wrapped_interval> r;

    wrapped_interval zero(s.t);
    zero.lower = 0;
    zero.upper = 0;

    for (auto &u : s.ssplit())
      for (auto &v : t.ssplit())
      {
        // Only optimize if its singleton
        auto v_non_zero = difference(v, zero);
        if ((u / v_non_zero).cardinality() == 1)
        {
          r.push_back(u - ((u / v_non_zero) * v_non_zero));
          continue;
        }
        if (is_signedbv_type(u.t) && s.most_significant_bit(*u.upper))
          r.push_back(-amb(v));
        else
          r.push_back(amb(v));
      }

    return over_join(r);
  }

  friend wrapped_interval
  operator/(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    // [a_0, a_1] + [b_0, b_1] = [a_0+b_0, a_1 + b_1]
    std::vector<wrapped_interval> r;

    wrapped_interval zero(lhs.t);
    zero.lower = 0;
    zero.upper = 0;
    if (!is_signedbv_type(lhs.t))
    {
      for (const auto &u : lhs.ssplit())
        for (const auto &v : rhs.ssplit())
        {
          auto non_zero = difference(v, zero);
          wrapped_interval temp(lhs.t);
          temp.lower = *u.lower / *non_zero.upper;
          temp.upper = *u.upper / *non_zero.lower;
          r.push_back(temp);
        }
    }
    else
    {
      for (const auto &u : cut(lhs))
        for (const auto &v : cut(rhs))
        {
          auto non_zero = difference(v, zero);
          wrapped_interval temp(lhs.t);

          auto msb_a = temp.most_significant_bit(*u.lower);
          auto msb_c = temp.most_significant_bit(*non_zero.lower);

          if (!msb_a && !msb_c)
          {
            temp.set_lower(u.get_lower() / non_zero.get_upper());
            temp.set_upper(u.get_upper() / non_zero.get_lower());
          }
          else if (msb_a && msb_c)
          {
            temp.set_lower(u.get_upper() / non_zero.get_lower());
            temp.set_upper(u.get_lower() / non_zero.get_upper());
          }
          else if (!msb_a && msb_c)
          {
            temp.set_lower(u.get_upper() / non_zero.get_upper());
            temp.set_upper(u.get_lower() / non_zero.get_lower());
          }
          else if (msb_a && !msb_c)
          {
            temp.set_lower(u.get_lower() / non_zero.get_lower());
            temp.set_upper(u.get_upper() / non_zero.get_upper());
          }
          else
          {
            log_error("This should never happen");
            abort();
          }

          r.push_back(temp);
        }
    }
    return over_join(r);
  }

  typedef std::function<uint64_t(uint64_t, uint64_t, uint64_t, uint64_t)>
    warren_approximation_function;

  friend wrapped_interval
  operator|(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    const unsigned width = lhs.t->get_width();
    const auto min_or =
      [&width](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        uint64_t m, temp;
        m = compute_m(width);
        while (m != 0)
        {
          if (~a & c & m)
          {
            temp = (a | m) & -m;
            if (temp <= b)
            {
              a = temp;
              break;
            }
          }
          else if (a & ~c & m)
          {
            temp = (c | m) & -m;
            if (temp <= d)
            {
              c = temp;
              break;
            }
          }
          m = m >> 1;
        }
        return a | c;
      };
    const auto max_or =
      [&width](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        uint64_t m, temp;
        m = compute_m(width);
        if (width == 32)
          assert(m == 0x80000000);
        while (m != 0)
        {
          if (b & d & m)
          {
            temp = (b - m) | (m - 1);
            if (temp >= a)
            {
              b = temp;
              break;
            }
            temp = (d - m) | (m - 1);
            if (temp >= c)
            {
              d = temp;
              break;
            }
          }
          m = m >> 1;
        }
        return b | d;
      };
    return warren_approximation(lhs, rhs, min_or, max_or);
  }

  friend wrapped_interval
  operator&(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    const unsigned width = lhs.t->get_width();
    const auto min_and =
      [&width](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        uint64_t m, temp;

        m = compute_m(width);
        while (m != 0)
        {
          if (~a & ~c & m)
          {
            temp = (a | m) & -m;
            if (temp <= b)
            {
              a = temp;
              break;
            }
            temp = (c | m) & -m;
            if (temp <= d)
            {
              c = temp;
              break;
            }
          }
          m = m >> 1;
        }
        return a & c;
      };
    const auto max_and =
      [&width](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        uint64_t m, temp;

        m = compute_m(width);
        while (m != 0)
        {
          if (b & ~d & m)
          {
            temp = (b & ~m) | (m - 1);
            if (temp >= a)
            {
              b = temp;
              break;
            }
          }
          else if (~b & d & m)
          {
            temp = (d & ~m) | (m - 1);
            if (temp >= c)
            {
              d = temp;
              break;
            }
          }
          m = m >> 1;
        }
        return b & d;
      };
    return warren_approximation(lhs, rhs, min_and, max_and);
  }

  friend wrapped_interval
  operator^(const wrapped_interval &lhs, const wrapped_interval &rhs)
  {
    const unsigned width = lhs.t->get_width();
    const auto min_xor =
      [&width](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        uint64_t m, temp;
        m = compute_m(width);
        while (m != 0)
        {
          if (~a & c & m)
          {
            temp = (a | m) & -m;
            if (temp <= b)
            {
              a = temp;
            }
          }
          else if (a & ~c & m)
          {
            temp = (c | m) & -m;
            if (temp <= d)
            {
              c = temp;
            }
          }
          m = m >> 1;
        }
        return a ^ c;
      };
    const auto max_xor =
      [&width](uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        uint64_t m, temp;
        m = compute_m(width);
        while (m != 0)
        {
          if (b & d & m)
          {
            temp = (b - m) | (m - 1);
            if (temp >= a)
            {
              b = temp;
            }
            else
            {
              temp = (d - m) | (m - 1);
              if (temp >= c)
              {
                d = temp;
              }
            }
          }
          m = m >> 1;
        }
        return b ^ d;
      };
    return warren_approximation(lhs, rhs, min_xor, max_xor);
  }

  static wrapped_interval bitnot(const wrapped_interval &w)
  {
    wrapped_interval result(w.t);
    result.set_lower(-w.get_upper() - 1);
    result.set_upper(-w.get_lower() - 1);
    return result;
  }

  const BigInt &get_upper_bound() const
  {
    return upper_bound;
  }

  type2tc t;

  static wrapped_interval ternary_if(
    const wrapped_interval &cond,
    const wrapped_interval &true_value,
    const wrapped_interval &false_value)
  {
    if (!cond.contains(0))
      return true_value;

    if (cond.singleton())
      return false_value;

    return over_join(true_value, false_value);
  }

protected:
  BigInt upper_bound;

private:
  static BigInt compute_upper_bound(const type2tc &t)
  {
    if (is_pointer_type(t))
      return BigInt::power2(config.ansi_c.address_width);

    return BigInt::power2(t->get_width());
  }

  // For bitwise intervals approximations (Warren 2002)
  static uint64_t compute_m(unsigned width)
  {
    assert(width - 1 < 64);
    uint64_t m = (uint64_t)1 << (width - 1);
    if (width == 32)
      assert(m == 0x80000000);
    return m;
  }

  static wrapped_interval warren_approximation(
    const wrapped_interval &lhs,
    const wrapped_interval &rhs,
    const warren_approximation_function &min,
    const warren_approximation_function &max)
  {
    // TODO: BigInt has no support for bitwise operators
    wrapped_interval result(lhs.t);
    if (lhs.t->get_width() > 64)
      return result;

    std::vector<wrapped_interval> r;
    for (auto &u : lhs.ssplit())
      for (auto &v : rhs.ssplit())
      {
        auto u_lower = (*u.lower).to_uint64();
        auto u_upper = (*u.upper).to_uint64();

        auto v_lower = (*v.lower).to_uint64();
        auto v_upper = (*v.upper).to_uint64();

        wrapped_interval temp(lhs.t);
        temp.lower = min(u_lower, u_upper, v_lower, v_upper);
        temp.upper = max(u_lower, u_upper, v_lower, v_upper);
        r.push_back(temp);
      }

    return over_join(r);
  }
};
