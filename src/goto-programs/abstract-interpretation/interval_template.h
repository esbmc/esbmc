#ifndef CPROVER_ANALYSES_INTERVAL_TEMPLATE_H
#define CPROVER_ANALYSES_INTERVAL_TEMPLATE_H

#include <algorithm>
#include <iosfwd>
#include <util/threeval.h>
#include <util/message.h>
#include <sstream>
#include <util/ieee_float.h>
#include <optional>
/**
 * @brief This class is used to store intervals
 * in the form of lower <= upper. It also has support
 * for infinite. For this the variables `_set` are used.
 * 
 * @tparam T template type for the numeric value (e.g. int, float, bigint)
 */
template <class T>
class interval_templatet
{
public:
  virtual ~interval_templatet() = default;

  interval_templatet() = default;

  explicit interval_templatet(const T &x) : lower(x), upper(x)
  {
  }

  interval_templatet(const T &l, const T &u) : lower(l), upper(u)
  {
  }

  /// Bound value
  std::optional<T> lower, upper;
  /// Type to be used for shift operations
  type2tc type;
  T get_lower() const
  {
    return get(false);
  }

  T get_upper() const
  {
    return get(true);
  }

  T adjust(const T &v) const
  {
    // This is just an id function
    return v;
  }

  virtual void set(bool is_upper, const T &b)
  {
    if (is_upper)
      upper = b;
    else
      lower = b;
  }

  virtual T get(bool is_upper) const
  {
    return is_upper ? *upper : *lower;
  }

  void set_upper(const T &b)
  {
    set(true, b);
  }
  void set_lower(const T &b)
  {
    set(false, b);
  }

  void dump() const
  {
    std::ostringstream oss;

    if (is_bottom())
      oss << "EMPTY";
    else
    {
      if (lower)

        oss << "[" << get_lower();
      else
        oss << "(-inf";

      oss << ",";
      if (upper)
        oss << get_upper() << "]";
      else
        oss << "+inf)";
    }
    log_status("{}", oss.str());
  }

  virtual bool is_le_than(const T &a, const T &b) const
  {
    return a <= b;
  }

  /**
 * @brief Checks whether there are values that satisfy the
 * the interval.
 */
  virtual bool empty() const
  {
    return lower && upper && *lower > *upper;
  }

  bool is_bottom() const // equivalent to 'false'
  {
    return empty();
  }

  virtual bool is_top() const // equivalent to 'true'
  {
    return !lower && !upper;
  }

  /// There is only one value that satisfies
  bool singleton() const
  {
    return upper && lower && *lower == *upper;
  }

  // constraints
  virtual void make_le_than(const T &v) // add upper bound
  {
    auto value = adjust(v);
    if (upper)
    {
      if (is_le_than(value, get_upper()))
        set_upper(value);
    }
    else
    {
      set_upper(value);
    }
  }

  virtual bool contains(const T &e) const
  {
    if (is_top())
      return true;

    if (lower && e < get_lower())
      return false;

    if (upper && e > get_upper())
      return false;

    return true;
  }

  // Sound version (considering over approximations)
  void make_sound_le(interval_templatet<T> &v)
  {
    // [lower, upper] <= [v.lower,v.upper] <==> [lower, min(upper,v.upper)] <= [max(lower, v.lower), v.upper]

    if (upper || v.upper)
      upper = upper && v.upper ? std::min(*upper, *v.upper)
              : upper          ? *upper
                               : *v.upper;
    if (lower || v.lower)
      v.lower = lower && v.lower ? std::max(*lower, *v.lower)
                : lower          ? *lower
                                 : *v.lower;
  }

  virtual void make_ge_than(const T &v) // add lower bound
  {
    auto value = adjust(v);
    if (lower)
    {
      if (is_le_than(get_lower(), value))
        set_lower(value);
    }
    else
    {
      set_lower(value);
    }
  }

  /// Union or disjunction
  void join(const interval_templatet<T> &i)
  {
    approx_union_with(i);
  }

  /// Intersection or conjunction
  void meet(const interval_templatet<T> &i)
  {
    intersect_with(i);
  }

  void intersect_with(const interval_templatet &i)
  {
    if (i.lower)
    {
      if (lower)
      {
        lower = std::max(*lower, *i.lower);
      }
      else
      {
        lower = *i.lower;
      }
    }

    if (i.upper)
    {
      if (upper)
      {
        upper = std::min(*upper, *i.upper);
      }
      else
      {
        upper = *i.upper;
      }
    }
  }

  virtual void approx_union_with(const interval_templatet &i)
  {
    if (i.lower && lower)
      lower = std::min(*lower, *i.lower);
    else if (!i.lower)
      lower.reset();

    if (i.upper && upper)
      upper = std::max(*upper, *i.upper);
    else if (!i.upper)
      upper.reset();
  }

  /* INTERVAL ARITHMETICS 
   *
   * Following Chapter 32 of Principles of Abstract Interpretation.
   * 
   * ADD/SUB
   * [x0, x1] + empty <==>  empty + [x0, x1] <==> [x0, x1] - empty <==>  empty + [x0, x1] <==> empty
   * [x0, x1] + [y0, y1] <==> [x0+y0, x1+y1]
   * [x0, x1] - [y0, y1] <==> [x0-y1, x1-y0]
   * -[x0, x1] <==> [-x1, -x0]
   * -infinity + infinity <==> -infinity
   * -infinity + c <==> -infinity
   * infinity + infinity = infinity
   * 
   * MULT
   * [x0, x1] * empty <==> empty * [x0, x1] <==> empty
   * [x0, x1] * [y0, y1] <==> [min(x0*y0, x0*y1, x1*y0, x1*y1), max(x0*y0, x0*y1, x1*y0, x1*y1)]
   * 
   * DIV
   * [1, 1] / [x0, x1] <==> {
   *  (x1 < 0) || (0 < x0) => [1/x1, 1/x0]
   *  otherwise (e.g. 1/0) => [-infinity, infinity]
   *  }
   * [x0, x1] / [y0, y1] <==> [x0, x1] * [[1,1] / [y0, y1]]
   * 
   * 
  */
  friend interval_templatet<T>
  operator+(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    // [a_0, a_1] + [b_0, b_1] = [a_0+b_0, a_1 + b_1]
    auto result = rhs.empty() ? rhs : lhs;
    if (result.empty())
      return result;

    if (!lhs.lower || !rhs.lower)
      result.lower.reset();
    else
      result.lower = *lhs.lower + *rhs.lower;

    if (!lhs.upper || !rhs.upper)
      result.upper.reset();
    else
      result.upper = *lhs.upper + *rhs.upper;

    return result;
  }

  friend interval_templatet<T> operator-(const interval_templatet<T> &lhs)
  {
    // -[a_0, a_1] = [-a_1, -a_0]
    auto result = lhs;
    if (!lhs.upper)
      result.lower.reset();
    else
    {
      result.lower = -(*lhs.upper);
    }

    if (!lhs.lower)
    {
      result.upper.reset();
    }
    else
    {
      result.upper = -(*lhs.lower);
    }
    return result;
  }

  friend interval_templatet<T>
  operator-(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    // [a_0, a_1] - [b_0, b_1] = [a_0-b_1, a_1 - b_0]
    auto result = rhs.empty() ? rhs : lhs;
    if (result.empty())
      return result;

    if (!lhs.lower || !rhs.upper)
      result.lower.reset();
    else
      result.lower = *lhs.lower - *rhs.upper;

    if (!lhs.upper || !rhs.lower)
      result.upper.reset();
    else
      result.upper = *lhs.upper - *rhs.lower;

    return result;
  }

  friend interval_templatet<T>
  operator*(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    // [a_0, a_1] * [b_0, b_1] = [min(a_0*b_0, a_0*b_1, a_1*b_0, a_1*b_1), max(a_0*b_0, a_0*b_1, a_1*b_0, a_1*b_1)]
    interval_templatet<T> result;
    if (rhs.empty() || lhs.empty())
      return rhs.empty() ? rhs : lhs;

    // Let's deal with infinities first
    if (!lhs.lower || !rhs.lower || !lhs.upper || !rhs.upper)
      return result;

    // Initialize with a0 * b0
    auto a0_b0 = *lhs.lower * *rhs.lower;
    result.lower = a0_b0;
    result.upper = a0_b0;

    auto update_value = [&result](T value) {
      result.lower = std::min(value, *result.lower);
      result.upper = std::max(value, *result.upper);
    };

    update_value(*lhs.lower * *rhs.upper); // a0 * b1
    update_value(*lhs.upper * *rhs.lower); // a1 * b0
    update_value(*lhs.upper * *rhs.upper); // a1 * b1

    return result;
  }

  friend interval_templatet<T>
  operator/(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    /* Note, some works suggests doing a multiplication as [a0, a_1] * ([1,1] / [b_0, b_1])
     * However, our implementation is not a symbolic computation; thus the arithmetic is not
     * associative. For example, [-6, 10] / 2 which is [-3, 5] cannot be computed through
     * the [-6, 10] * [1/2, 1/2] because 1/2 will result in 0.
    */
    if (rhs.empty() || lhs.empty())
      return rhs.empty() ? rhs : lhs;

    interval_templatet<T> result;

    // Let's (not) deal with infinities first and division by 0.
    if (
      !lhs.lower || !rhs.lower || !lhs.upper || !rhs.upper || *rhs.lower == 0 ||
      *rhs.upper == 0)
      return result;

    // Initialize with a0 * b0
    T a0_b0 = *lhs.lower / *rhs.lower;
    result.lower = a0_b0;
    result.upper = a0_b0;

    auto update_value = [&result](T value) {
      result.lower = std::min(value, *result.lower);
      result.upper = std::max(value, *result.upper);
    };

    update_value(*lhs.lower / *rhs.upper); // a0 / b1
    update_value(*lhs.upper / *rhs.lower); // a1 / b0
    update_value(*lhs.upper / *rhs.upper); // a1 / b1

    return result;
  }

  static interval_templatet<T> ternary_if(
    const interval_templatet<T> &cond,
    const interval_templatet<T> &true_value,
    const interval_templatet<T> &false_value)
  {
    if (!cond.contains(0))
      return true_value;

    if (cond.singleton())
      return false_value;

    interval_templatet<T> result = true_value;
    result.join(false_value);
    return result;
  }

  static interval_templatet<T>
  cast(const interval_templatet<T> &, const type2tc &)
  {
    log_debug("interval", "No support for typecasting");
    interval_templatet<T> result;
    return result;
  }

  friend interval_templatet<T>
  operator%(const interval_templatet<T> &, const interval_templatet<T> &)
  {
    log_debug("interval", "No support for mod");
    interval_templatet<T> result;
    return result;
  }

  interval_templatet<T> interval_bitand(
    const interval_templatet<T> &,
    const interval_templatet<T> &) const
  {
    log_debug("interval", "No support for bitand");
    interval_templatet<T> result;
    return result;
  }

  friend interval_templatet<T>
  operator&(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    return lhs.interval_bitand(lhs, rhs);
  }

  interval_templatet<T> interval_bitor(
    const interval_templatet<T> &,
    const interval_templatet<T> &) const
  {
    log_debug("interval", "No support for bitor");
    interval_templatet<T> result;
    return result;
  }

  friend interval_templatet<T>
  operator|(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    return lhs.interval_bitor(lhs, rhs);
  }

  interval_templatet<T> interval_bitxor(
    const interval_templatet<T> &,
    const interval_templatet<T> &) const
  {
    log_debug("interval", "No support for bitxor");
    interval_templatet<T> result;
    return result;
  }

  friend interval_templatet<T>
  operator^(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    return lhs.interval_bitxor(lhs, rhs);
  }

  interval_templatet<T> interval_logical_right_shift(
    const interval_templatet<T> &,
    const interval_templatet<T> &) const
  {
    log_debug("interval", "No support for lshr");
    interval_templatet<T> result;
    return result;
  }

  static interval_templatet<T> logical_right_shift(
    const interval_templatet<T> &lhs,
    const interval_templatet<T> &rhs)
  {
    return lhs.interval_logical_right_shift(lhs, rhs);
  }

  interval_templatet<T> interval_left_shift(
    const interval_templatet<T> &,
    const interval_templatet<T> &) const
  {
    log_debug("interval", "No support for shl");
    interval_templatet<T> result;
    return result;
  }

  static interval_templatet<T>
  left_shift(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    return lhs.interval_left_shift(lhs, rhs);
  }

  static interval_templatet<T> arithmetic_right_shift(
    const interval_templatet<T> &lhs,
    const interval_templatet<T> &rhs)
  {
    return lhs.interval_left_shift(lhs, rhs);
  }

  static interval_templatet<T> bitnot(const interval_templatet<T> &w)
  {
    interval_templatet<T> result;
    return result;
  }

  static interval_templatet<T>
  equality(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    interval_templatet<T> result;
    result.set_lower(0);
    result.set_upper(1);

    // If the intervals are singletons and equal then it's always true
    if (
      lhs.singleton() && rhs.singleton() && lhs.get_lower() == rhs.get_lower())
    {
      result.set_lower(1);
      return result;
    }

    // If the intervals don't intersect, then they are always empty
    auto lhs_cpy = lhs;
    lhs_cpy.intersect_with(rhs);
    if (lhs_cpy.empty())
      result.set_upper(0);

    return result;
  }

  static interval_templatet<T> invert_bool(const interval_templatet<T> &i)
  {
    interval_templatet<T> result;
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

  static interval_templatet<T>
  not_equal(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    return invert_bool(equality(lhs, rhs));
  }

  static interval_templatet<T>
  less_than(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    interval_templatet<T> result;
    result.set_lower(0);
    result.set_upper(1);

    // MAX LHS < MIN RHS => TRUE
    if (lhs.upper && rhs.lower && lhs.get_upper() < rhs.get_lower())
    {
      result.set_lower(1);
      return result;
    }

    // MIN LHS >= MAX RHS => FALSE
    if (lhs.lower && rhs.upper && lhs.get_lower() >= rhs.get_upper())
    {
      result.set_upper(0);
      return result;
    }

    assert(!result.singleton() && !result.is_bottom());
    return result;
  }

  static interval_templatet<T> less_than_equal(
    const interval_templatet<T> &lhs,
    const interval_templatet<T> &rhs)
  {
    interval_templatet<T> result;
    result.set_lower(0);
    result.set_upper(1);

    // MAX LHS <= MIN RHS => TRUE
    if (lhs.upper && rhs.lower && lhs.get_upper() <= rhs.get_lower())
    {
      result.set_lower(1);
      return result;
    }

    // MIN LHS > MAX RHS => FALSE
    if (lhs.lower && rhs.upper && lhs.get_lower() > rhs.get_upper())
    {
      result.set_upper(0);
      return result;
    }

    assert(!result.singleton() && !result.is_bottom());
    return result;
  }

  static interval_templatet<T> greater_than_equal(
    const interval_templatet<T> &lhs,
    const interval_templatet<T> &rhs)
  {
    // a >= b <==> b <= a
    return less_than_equal(rhs, lhs);
  }

  static interval_templatet<T> greater_than(
    const interval_templatet<T> &lhs,
    const interval_templatet<T> &rhs)
  {
    // a > b <==> b < a
    return less_than(rhs, lhs);
  }

  /// This is just to check if a value has changed. This is not the same as an interval comparation!
  bool inline has_changed(const interval_templatet<T> &i)
  {
    if (empty())
      return false;

    if ((lower != i.lower) || (upper != i.upper))
      return true;

    return false;
  }

  /// Generates an interval of the form (-infinity, 0]
  void make_lower_interval()
  {
    // [-infinity, 0]
    make_le_than(0);
  }

  /**
     * @brief Computer the contraction of "a" and "b" under a <= b
     * 
     * @param a 
     * @param b 
     */
  static void
  contract_interval_le(interval_templatet<T> &a, interval_templatet<T> &b)
  {
    /**
       * 1. Forward Evaluation y = (a - b) ===> [y] = ([a] - [b]) intersect [-infinity, 0]
       * 2. Backwards Step, for each variable:
       *   a. [a] = [a] intersect [b] + [y]
       *   b. [b] = [b] intersect [a] - [y]
       * 3. Find a fixpoint. 
       * 
       */
    interval_templatet<T> intersection_operand;
    intersection_operand.make_lower_interval();
    bool changed;
    do
    {
      changed = false;
      auto tmp_a = a;
      auto tmp_b = b;
      auto y = (a - b);
      y.intersect_with(intersection_operand);

      a.intersect_with(tmp_b + y);
      b.intersect_with(tmp_a - y);

      changed = a.has_changed(tmp_a) || b.has_changed(tmp_b);
    } while (changed);
  }
};

template <class T>
tvt operator<=(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  if (a.upper && b.lower && *a.upper <= *b.lower)
    return tvt(true);
  if (a.lower && b.upper && *a.lower > *b.upper)
    return tvt(false);

  return tvt(tvt::TV_UNKNOWN);
}

template <class T>
tvt operator>=(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  return b <= a;
}

template <class T>
tvt operator<(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  return !(a >= b);
}

template <class T>
tvt operator>(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  return !(a <= b);
}

template <class T>
bool operator==(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  if (a.lower.has_value() != b.lower.has_value())
    return false;
  if (a.upper.has_value() != b.upper.has_value())
    return false;

  if (a.lower && *a.lower != *b.lower)
    return false;
  if (a.upper && *a.upper != *b.upper)
    return false;

  return true;
}

template <class T>
bool operator!=(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  return !(a == b);
}

template <class T>
interval_templatet<T> upper_interval(const T &u)
{
  interval_templatet<T> i;
  i.upper = u;
  return i;
}

template <class T>
interval_templatet<T> lower_interval(const T &l)
{
  interval_templatet<T> i;
  i.lower = l;
  return i;
}

template <class T>
std::ostream &operator<<(std::ostream &out, const interval_templatet<T> &i)
{
  if (i.lower)
    out << '[' << *i.lower;
  else
    out << ")-INF";

  out << ',';

  if (i.upper)
    out << *i.upper << ']';
  else
    out << "+INF(";

  return out;
}
#endif // CPROVER_ANALYSES_INTERVAL_TEMPLATE_H
