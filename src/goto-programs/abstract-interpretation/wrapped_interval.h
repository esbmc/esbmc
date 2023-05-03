#pragma once

#include "interval_template.h"

class wrapped_interval : public interval_templatet<BigInt>
{
public:
  wrapped_interval(const type2tc &t) : t(t)
  {
    assert(is_signedbv_type(t) || is_unsignedbv_type(t));
    lower_set = true;
    upper_set = true;

    lower = 0;
    upper = get_upper_bound(t);
  }

  static BigInt get_upper_bound(const type2tc &t) {
    assert(is_signedbv_type(t) || is_unsignedbv_type(t));
    BigInt r(1);
    r.setPower2(t->get_width()*8);
    return r;
  }

  const type2tc &t;
  bool bottom = false;
  virtual bool empty() const override
  {
    return bottom;
  }

  virtual bool is_top() const override
  {
    return !empty() && lower == 0 && upper == get_upper_bound(t);
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
    auto lhs = (b-a)%mod;
    if(lhs < 0)
      lhs = lhs + mod;
    auto rhs = (c-a)%mod;
    if(rhs < 0)
      rhs = rhs+mod;

    return lhs <= rhs;
  }

  BigInt cardinality() const {
    if(is_bottom()) return 0;
    if(is_top()) return get_upper_bound(t);

    auto mod = get_upper_bound(t);
    return ((upper-lower)%mod+1)%mod;
  }

  bool is_member(const BigInt &e) const {
    if(is_top()) return true;

    return wrapped_le(e, lower, upper, t);
  }

  bool is_equal(const wrapped_interval &rhs) const {
    return t == rhs.t && lower == rhs.lower && upper == rhs.upper;
  }

  bool is_included(const wrapped_interval &rhs) const {
    if(is_bottom() || rhs.is_top() || is_equal(rhs)) return true;
    if(is_top() || rhs.is_bottom()) return false;

    return rhs.is_member(lower) && rhs.is_member(upper) && (!is_member(rhs.lower) || !is_member(rhs.upper));
  }

  /// Over union
  static wrapped_interval over_join(const wrapped_interval &s, const wrapped_interval &t)
  {
    assert(s.t == t.t);

    if(s.is_included(t)) return t;
    if(t.is_included(s)) return s;

    const BigInt a = s.lower;
    const BigInt b = s.upper;
    const BigInt c = t.lower;
    const BigInt d = t.upper;

    wrapped_interval result(s.t);

    if(t.is_member(a) && t.is_member(b) && s.is_member(c) && s.is_member(d))
      return result;

    if(t.is_member(b) && s.is_member(c))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    if(s.is_member(d) && t.is_member(a))
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
  static wrapped_interval under_meet(const wrapped_interval &s, const wrapped_interval &t)
  {
    return complement(over_join(complement(s), complement(t)));
  }
  /// Over meet
  static wrapped_interval over_meet(const wrapped_interval &s, const wrapped_interval &t)
  {
    // TODO: this will probably not be needed!
    assert(s.t == t.t);

    if(s.is_included(t)) return s;
    if(t.is_included(s)) return t;

    const BigInt a = s.lower;
    const BigInt b = s.upper;
    const BigInt c = t.lower;
    const BigInt d = t.upper;

    wrapped_interval result(s.t);

    if(!t.is_member(a) && !t.is_member(b) && !s.is_member(c) && !s.is_member(d))
    {
      result.bottom = true;
      return result;
    }


    if(s.is_member(d) && t.is_member(a))
    {
      result.lower = a;
      result.upper = d;
      return result;
    }

    if(t.is_member(b) && s.is_member(c)){
      result.lower = c;
      result.upper = b;
      return result;
    }


    if((s.cardinality() < t.cardinality()) || (s.cardinality() == t.cardinality() && a <= c))
    {
      return s;
    }

    result.lower = c;
    result.upper = d;
    return result;


  }
};
