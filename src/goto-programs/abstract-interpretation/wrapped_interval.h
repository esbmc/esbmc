#pragma once

#include "interval_template.h"

class wrapped_interval : public interval_templatet<BigInt>
{
public:
  wrapped_interval()
  {
  }

  explicit wrapped_interval(const type2tc &t) : t(t)
  {
    assert(is_signedbv_type(t) || is_unsignedbv_type(t));
    lower_set = true;
    upper_set = true;

    lower = 0;
    upper = get_upper_bound(t) - 1;
  }

  BigInt convert_to_wrap(const BigInt &b) const
  {
    auto is_signed = is_signedbv_type(t);
    auto value = b;
    if(is_signed)
    {
      assert(value >= (-get_upper_bound(t) / 2));
      assert(value <= (get_upper_bound(t) / 2 - 1));
      value += get_upper_bound(t) / 2;
    }
    else
    {
      assert(value >= 0);
      assert(value <= get_upper_bound(t));
    }

    return value;
  }

  void set(bool is_upper, const BigInt &b) override
  {
    auto value = convert_to_wrap(b);

    if(is_upper)
      upper = value;
    else
      lower = value;
  }

  BigInt get(bool is_upper) const override
  {
    auto is_signed = is_signedbv_type(t);
    auto value = is_upper ? upper : lower;

    if(is_signed)
      value -= get_upper_bound(t) / 2;

    return value;
  }

  static BigInt get_upper_bound(const type2tc &t)
  {
    assert(is_signedbv_type(t) || is_unsignedbv_type(t));
    BigInt r(1);
    r.setPower2(t->get_width() * 8);
    return r;
  }

  type2tc t;
  bool bottom = false;
  virtual bool empty() const override
  {
    return bottom;
  }

  virtual bool is_top() const override
  {
    return !empty() && lower == 0 && upper == (get_upper_bound(t) - 1);
  }

  static wrapped_interval complement(const wrapped_interval &w) {
    wrapped_interval result(w.t);

    if(w.is_top())
      result.bottom = true;

    if(!w.is_top() && !w.is_bottom())
    {
      auto mod = get_upper_bound(w.t);
      result.lower = (w.upper + 1)%mod;
      result.upper = (w.lower - 1)%mod;
      if(result.upper < 0)
        result.upper += mod;
    }

    return result;
  }

  /// b <=_a c iff b -_w a <= c -_w a
  static bool wrapped_le(const BigInt &b, const BigInt &a, const BigInt &c, const type2tc &t) {
    auto mod = get_upper_bound(t);
    // https://torstencurdt.com/tech/posts/modulo-of-negative-numbers/
    auto lhs = (b - a) % mod;
    if(lhs < 0)
      lhs = lhs + mod;
    auto rhs = (c - a) % mod;
    if(rhs < 0)
      rhs = rhs + mod;

    return lhs <= rhs;
  }

  virtual void make_le_than(const BigInt &v) override // add upper bound
  {
    wrapped_interval value(t);
    value.set_upper(v);

    *this = intersection(*this, value);
  }

  virtual void make_ge_than(const BigInt &v) override // add upper bound
  {
    wrapped_interval value(t);
    value.set_lower(v);

    *this = intersection(*this, value);
  }

  BigInt cardinality() const
  {
    if(is_bottom())
      return 0;
    if(is_top())
      return get_upper_bound(t);

    auto mod = get_upper_bound(t);
    return ((upper - lower) % mod + 1) % mod;
  }

  bool contains(const BigInt &e) const
  {
    if(is_top())
      return true;

    return wrapped_le(e, lower, upper, t);
  }

  bool is_equal(const wrapped_interval &rhs) const
  {
    return t == rhs.t && lower == rhs.lower && upper == rhs.upper;
  }

  bool is_included(const wrapped_interval &rhs) const
  {
    if(is_bottom() || rhs.is_top() || is_equal(rhs))
      return true;
    if(is_top() || rhs.is_bottom())
      return false;

    return rhs.contains(lower) && rhs.contains(upper) &&
           (!contains(rhs.lower) || !contains(rhs.upper));
  }

  /// Over union
  static wrapped_interval
  over_join(const wrapped_interval &s, const wrapped_interval &t)
  {
    assert(s.t == t.t);

    if(s.is_included(t)) return t;
    if(t.is_included(s)) return s;

    const BigInt a = s.lower;
    const BigInt b = s.upper;
    const BigInt c = t.lower;
    const BigInt d = t.upper;

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

    if((b_c.cardinality() < d_a.cardinality()) || (b_c.cardinality() == d_a.cardinality() && a <= c))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    result.lower = c;
    result.upper = b;
    return result;
  }

  // Under meet
  static wrapped_interval
  under_meet(const wrapped_interval &s, const wrapped_interval &t)
  {
    return complement(over_join(complement(s), complement(t)));
  }

  static wrapped_interval
  intersection(const wrapped_interval &s, const wrapped_interval &t)
  {
    assert(s.t == t.t);

    wrapped_interval result(s.t);

    if(s.is_bottom() || t.is_bottom())
    {
      result.bottom = true;
      return result;
    }

    if(s.is_equal(t) || s.is_top())
    {
      return t;
    }

    if(t.is_top())
    {
      return s;
    }

    const BigInt a = s.lower;
    const BigInt b = s.upper;
    const BigInt c = t.lower;
    const BigInt d = t.upper;

    if(t.contains(a) && t.contains(b) && s.contains(c) && s.contains(d))
    {
      wrapped_interval a_d(s.t);
      a_d.lower = a;
      a_d.upper = d;

      wrapped_interval c_b(s.t);
      c_b.lower = c;
      c_b.upper = b;
      // TODO: we can start working with double intervals (more memory though)
      // Note: We can overaproximate as long as we don't exclude real intersection values
      return over_join(a_d, c_b);
    }

    if(t.contains(a) && t.contains(b))
      return s;

    if(s.contains(c) && s.contains(d))
      return t;

    if(t.contains(a) && s.contains(d) && !t.contains(b) && !s.contains(c))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    if(t.contains(b) && s.contains(c) && !t.contains(a) && !s.contains(d))
    {
      result.lower = c;
      result.upper = b;
      return result;
    }

    result.bottom = true;
    return result;
  }

  /// Over meet
  static wrapped_interval
  over_meet(const wrapped_interval &s, const wrapped_interval &t)
  {
    // TODO: this will probably not be needed!
    assert(s.t == t.t);

    if(s.is_included(t))
      return s;
    if(t.is_included(s))
      return t;

    const BigInt a = s.lower;
    const BigInt b = s.upper;
    const BigInt c = t.lower;
    const BigInt d = t.upper;

    wrapped_interval result(s.t);

    if(!t.contains(a) && !t.contains(b) && !s.contains(c) && !s.contains(d))
    {
      result.bottom = true;
      return result;
    }

    if(s.contains(d) && t.contains(a))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    if(t.contains(b) && s.contains(c))
    {
      result.lower = c;
      result.upper = b;
      return result;
    }

    if(
      (s.cardinality() < t.cardinality()) ||
      (s.cardinality() == t.cardinality() && a <= c))
    {
      return s;
    }

    result.lower = c;
    result.upper = d;
    return result;
  }
};
