#ifndef CPROVER_ANALYSES_INTERVAL_TEMPLATE_H
#define CPROVER_ANALYSES_INTERVAL_TEMPLATE_H

#include <algorithm>
#include <iosfwd>
#include <util/threeval.h>
#include <util/message.h>
#include <sstream>
#include <util/ieee_float.h>
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
  interval_templatet() : lower_set(false), upper_set(false)
  {
    // this is 'top'
  }

  explicit interval_templatet(const T &x)
    : lower_set(true), upper_set(true), lower(x), upper(x)
  {
  }

  explicit interval_templatet(const T &l, const T &u)
    : lower_set(true), upper_set(true), lower(l), upper(u)
  {
  }

  /* TODO: There is a clear dependency between the lower/upper and
  *        and lower_set/upper_set variable. Shouldn't we convert the
  *        the uses into the constraints functions?
  */
  /// If the `_set` variables are false, then the bound is infinity
  bool lower_set, upper_set;
  /// Bound value
  T lower, upper;

  const T &get_lower() const
  {
    return lower;
  }

  const T &get_upper() const
  {
    return upper;
  }

  /**
 * @brief Checks whether there are values that satisfy the
 * the interval.
 */
  bool empty() const
  {
    return upper_set && lower_set && lower > upper;
  }

  bool is_bottom() const // equivalent to 'false'
  {
    return empty();
  }

  bool is_top() const // equivalent to 'true'
  {
    return !lower_set && !upper_set;
  }

  /// There is only one value that satisfies
  bool singleton() const
  {
    return upper_set && lower_set && lower == upper;
  }

  // constraints
  void make_le_than(const T &v) // add upper bound
  {
    if(upper_set)
    {
      if(upper > v)
        upper = v;
    }
    else
    {
      upper_set = true;
      upper = v;
    }
  }

  void make_ge_than(const T &v) // add lower bound
  {
    if(lower_set)
    {
      if(lower < v)
        lower = v;
    }
    else
    {
      lower_set = true;
      lower = v;
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
    if(i.lower_set)
    {
      if(lower_set)
      {
        lower = std::max(lower, i.lower);
      }
      else
      {
        lower_set = true;
        lower = i.lower;
      }
    }

    if(i.upper_set)
    {
      if(upper_set)
      {
        upper = std::min(upper, i.upper);
      }
      else
      {
        upper_set = true;
        upper = i.upper;
      }
    }
  }

  void approx_union_with(const interval_templatet &i)
  {
    if(i.lower_set && lower_set)
      lower = std::min(lower, i.lower);
    else if(!i.lower_set && lower_set)
      lower_set = false;

    if(i.upper_set && upper_set)
      upper = std::max(upper, i.upper);
    else if(!i.upper_set && upper_set)
      upper_set = false;
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
    if(result.empty())
      return result;

    if(!lhs.lower_set || !rhs.lower_set)
      result.lower_set = false;
    else
      result.lower = lhs.lower + rhs.lower;

    if(!lhs.upper_set || !rhs.upper_set)
      result.upper_set = false;
    else
      result.upper = lhs.upper + rhs.upper;

    return result;
  }

  friend interval_templatet<T> operator-(const interval_templatet<T> &lhs)
  {
    // -[a_0, a_1] = [-a_1, -a_0]
    auto result = lhs;
    if(!lhs.upper_set)
      result.lower_set = false;
    else
    {
      result.lower = -lhs.upper;
      result.lower_set = true;
    }

    if(!lhs.lower_set)
    {
      result.upper_set = false;
    }
    else
    {
      result.upper = -lhs.lower;
      result.upper_set = true;
    }
    return result;
  }

  friend interval_templatet<T>
  operator-(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    // [a_0, a_1] - [b_0, b_1] = [a_0-b_1, a_1 - b_0]
    auto result = rhs.empty() ? rhs : lhs;
    if(result.empty())
      return result;

    if(!lhs.lower_set || !rhs.upper_set)
      result.lower_set = false;
    else
      result.lower = lhs.lower - rhs.upper;

    if(!lhs.upper_set || !rhs.lower_set)
      result.upper_set = false;
    else
      result.upper = lhs.upper - rhs.lower;

    return result;
  }

  friend interval_templatet<T>
  operator*(const interval_templatet<T> &lhs, const interval_templatet<T> &rhs)
  {
    // [a_0, a_1] * [b_0, b_1] = [min(a_0*b_0, a_0*b_1, a_1*b_0, a_1*b_1), max(a_0*b_0, a_0*b_1, a_1*b_0, a_1*b_1)]
    interval_templatet<T> result;
    if(rhs.empty() || lhs.empty())
      return rhs.empty() ? rhs : lhs;

    // Let's deal with infinities first
    if(!lhs.lower_set || !rhs.lower_set || !lhs.upper_set || !rhs.upper_set)
      return result;

    result.lower_set = true;
    result.upper_set = true;

    // Initialize with a0 * b0
    auto a0_b0 = lhs.lower * rhs.lower;
    result.lower = a0_b0;
    result.upper = a0_b0;

    auto update_value = [&result](T value) {
      result.lower = std::min(value, result.lower);
      result.upper = std::max(value, result.upper);
    };

    update_value(lhs.lower * rhs.upper); // a0 * b1
    update_value(lhs.upper * rhs.lower); // a1 * b0
    update_value(lhs.upper * rhs.upper); // a1 * b1

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
    if(rhs.empty() || lhs.empty())
      return rhs.empty() ? rhs : lhs;

    interval_templatet<T> result;

    // Let's (not) deal with infinities first and division by 0.
    if(
      !lhs.lower_set || !rhs.lower_set || !lhs.upper_set || !rhs.upper_set ||
      rhs.lower == 0 || rhs.upper == 0)
      return result;

    result.lower_set = true;
    result.upper_set = true;

    // Initialize with a0 * b0
    auto a0_b0 = lhs.lower / rhs.lower;
    result.lower = a0_b0;
    result.upper = a0_b0;

    auto update_value = [&result](T value) {
      result.lower = std::min(value, result.lower);
      result.upper = std::max(value, result.upper);
    };

    update_value(lhs.lower / rhs.upper); // a0 / b1
    update_value(lhs.upper / rhs.lower); // a1 / b0
    update_value(lhs.upper / rhs.upper); // a1 / b1

    return result;
  }

  /// This is just to check if a value has changed. This is not the same as an interval comparation!
  bool inline has_changed(const interval_templatet<T> &i)
  {
    if(empty())
      return false;
    auto lower_equal = lower == i.lower;
    auto upper_equal = upper == i.upper;
    auto lower_set_equal = lower_set == i.lower_set;
    auto upper_set_equal = upper_set == i.upper_set;

    if(lower_set && upper_set)
      return !(
        lower_set_equal && upper_set_equal && lower_equal && upper_equal);

    if(lower_set)
      return !(upper_set_equal && upper_equal);

    if(upper_set)
      return !(lower_set_equal && lower_equal);

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

      std::ostringstream oss;
      oss << "[Contractor] Y: " << y;
      oss << "\n[Contractor] A: " << a;
      oss << "\n[Contractor] B: " << b;
      log_debug("\n\n{}", oss.str());
      changed = a.has_changed(tmp_a) || b.has_changed(tmp_b);
    } while(changed);
  }
};

template <class T>
tvt operator<=(const interval_templatet<T> &a, const interval_templatet<T> &b)
{
  if(a.upper_set && b.lower_set && a.upper <= b.lower)
    return tvt(true);
  if(a.lower_set && b.upper_set && a.lower > b.upper)
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
  if(a.lower_set != b.lower_set)
    return false;
  if(a.upper_set != b.upper_set)
    return false;

  if(a.lower_set && a.lower != b.lower)
    return false;
  if(a.upper_set && a.upper != b.upper)
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
  i.upper_set = true;
  i.upper = u;
  return i;
}

template <class T>
interval_templatet<T> lower_interval(const T &l)
{
  interval_templatet<T> i;
  i.lower_set = true;
  i.lower = l;
  return i;
}

template <class T>
std::ostream &operator<<(std::ostream &out, const interval_templatet<T> &i)
{
  if(i.lower_set)
    out << '[' << i.lower;
  else
    out << ")-INF";

  out << ',';

  if(i.upper_set)
    out << i.upper << ']';
  else
    out << "+INF(";

  return out;
}

template <>
void interval_templatet<ieee_floatt>::make_lower_interval();

template <>
bool interval_templatet<ieee_floatt>::is_top() const;

template <>
bool interval_templatet<const ieee_floatt>::is_top() const;

#endif // CPROVER_ANALYSES_INTERVAL_TEMPLATE_H
