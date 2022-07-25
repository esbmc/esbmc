#ifndef CPROVER_IEEE_FLOAT_H
#define CPROVER_IEEE_FLOAT_H

#include <util/format_spec.h>
#include <irep2/irep2_type.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/std_types.h>

class ieee_float_spect
{
public:
  unsigned f, e;

  BigInt bias() const;

  ieee_float_spect(const floatbv_typet &type);
  ieee_float_spect(const floatbv_type2tc &type);

  ieee_float_spect() : f(0), e(0)
  {
  }

  ieee_float_spect(unsigned _f, unsigned _e) : f(_f), e(_e)
  {
  }

  inline unsigned width() const
  {
    return f + e + 1;
  }

  const floatbv_type2tc get_type() const;

  BigInt max_exponent() const;
  BigInt max_fraction() const;

  floatbv_typet to_type() const;
  inline static ieee_float_spect single_precision()
  {
    // 32 bits in total
    return ieee_float_spect(23, 8);
  }

  inline static ieee_float_spect double_precision()
  {
    // 64 bits in total
    return ieee_float_spect(52, 11);
  }

  inline static ieee_float_spect quadruple_precision()
  {
    // IEEE 754 binary128
    return ieee_float_spect(112, 15);
  }

  inline friend bool
  operator==(const ieee_float_spect &a, const ieee_float_spect &b)
  {
    return a.f == b.f && a.e == b.e;
  }

  inline friend bool
  operator!=(const ieee_float_spect &a, const ieee_float_spect &b)
  {
    return !(a == b);
  }
};

bool operator==(const ieee_float_spect &a, const ieee_float_spect &b);
bool operator!=(const ieee_float_spect &a, const ieee_float_spect &b);

class ieee_floatt
{
public:
  typedef enum
  {
    ROUND_TO_EVEN = 0,
    ROUND_TO_AWAY = 1,
    ROUND_TO_PLUS_INF = 2,
    ROUND_TO_MINUS_INF = 3,
    ROUND_TO_ZERO = 4,
    UNKNOWN,
    NONDETERMINISTIC
  } rounding_modet;

  rounding_modet rounding_mode;
  ieee_float_spect spec;

  explicit ieee_floatt();
  explicit ieee_floatt(const ieee_float_spect &s);
  explicit ieee_floatt(const constant_exprt &expr);

  void negate()
  {
    sign_flag = !sign_flag;
  }

  void set_sign(bool _sign)
  {
    sign_flag = _sign;
  }

  void make_zero()
  {
    sign_flag = false;
    exponent = 0;
    fraction = 0;
    NaN_flag = false;
    infinity_flag = false;
  }

  void make_NaN();
  void make_plus_infinity();
  void make_minus_infinity();
  void make_fltmax(); // maximum representable finite floating-point number
  void make_fltmin(); // minimum normalized positive floating-point number

  static ieee_floatt NaN(const ieee_float_spect &_spec)
  {
    ieee_floatt c(_spec);
    c.make_NaN();
    return c;
  }

  static ieee_floatt plus_infinity(const ieee_float_spect &_spec)
  {
    ieee_floatt c(_spec);
    c.make_plus_infinity();
    return c;
  }

  static ieee_floatt minus_infinity(const ieee_float_spect &_spec)
  {
    ieee_floatt c(_spec);
    c.make_minus_infinity();
    return c;
  }

  // maximum representable finite floating-point number
  static ieee_floatt fltmax(const ieee_float_spect &_spec)
  {
    ieee_floatt c(_spec);
    c.make_fltmax();
    return c;
  }

  // minimum normalized positive floating-point number
  static ieee_floatt fltmin(const ieee_float_spect &_spec)
  {
    ieee_floatt c(_spec);
    c.make_fltmin();
    return c;
  }

  // set to next representable number towards plus infinity
  void increment(bool distinguish_zero = false)
  {
    if(is_zero() && get_sign() && distinguish_zero)
      negate();
    else
      next_representable(true);
  }

  // set to previous representable number towards minus infinity
  void decrement(bool distinguish_zero = false)
  {
    if(is_zero() && !get_sign() && distinguish_zero)
      negate();
    else
      next_representable(false);
  }

  bool is_zero() const
  {
    return !NaN_flag && !infinity_flag && fraction == 0 && exponent == 0;
  }

  bool is_negative() const
  {
    return sign_flag;
  }

  bool get_sign() const
  {
    return sign_flag;
  }

  bool is_NaN() const
  {
    return NaN_flag;
  }

  bool is_infinity() const
  {
    return !NaN_flag && infinity_flag;
  }

  bool is_finite() const
  {
    return !(infinity_flag && NaN_flag);
  }

  bool is_normal() const;

  const BigInt &get_exponent() const
  {
    return exponent;
  }

  const BigInt &get_fraction() const
  {
    return fraction;
  }

  // performs conversion to ieee floating point format
  void from_integer(const BigInt &i);
  void from_base10(const BigInt &exp, const BigInt &frac);
  void build(const BigInt &exp, const BigInt &frac);

  /* Given i non-negative, this method replaces the contents of this instance
   * by interpreting it according to the IEEE-754 in-memory representation.
   * In particular, the low-order spec.f bits are taken as the fraction, the
   * following spec.e bits as the exponent and the sign_flag is determined
   * based on the remaining bits being zero or not. Afterwards adjustments for
   * bias, denormalized numbers and NaNs are performed. */
  void unpack(const BigInt &i);

  void from_double(const double d);
  void from_float(const float f);
  double to_double() const;
  float to_float() const;
  bool is_double() const;
  bool is_float() const;
  BigInt pack() const;
  void extract_base2(BigInt &_fraction, BigInt &_exponent) const;
  void extract_base10(BigInt &_fraction, BigInt &_exponent) const;
  BigInt to_integer() const; // this always rounds to zero

  // performs conversion from ieee floating point format
  void change_spec(const ieee_float_spect &dest_spec);

  void print(std::ostream &out) const;

  std::string to_ansi_c_string() const
  {
    return format(format_spect());
  }

  std::string to_string_decimal(unsigned precision) const;
  std::string to_string_scientific(unsigned precision) const;
  std::string format(const format_spect &format_spec) const;

  friend inline std::ostream &
  operator<<(std::ostream &out, const ieee_floatt &f)
  {
    return out << f.to_ansi_c_string();
  }
  // expressions
  constant_exprt to_expr() const;
  void from_expr(const constant_exprt &expr);

  ieee_floatt &operator/=(const ieee_floatt &other);
  ieee_floatt &operator*=(const ieee_floatt &other);
  ieee_floatt &operator+=(const ieee_floatt &other);
  ieee_floatt &operator-=(const ieee_floatt &other);

  ieee_floatt &operator-();

  friend bool operator<(const ieee_floatt &a, const ieee_floatt &b);
  friend bool operator<=(const ieee_floatt &a, const ieee_floatt &b);
  friend bool operator>(const ieee_floatt &a, const ieee_floatt &b);
  friend bool operator>=(const ieee_floatt &a, const ieee_floatt &b);

  // warning: these do packed equality, not IEEE equality
  // e.g., NAN==NAN
  friend bool operator==(const ieee_floatt &a, const ieee_floatt &b);
  friend bool operator!=(const ieee_floatt &a, const ieee_floatt &b);

  friend bool operator==(const ieee_floatt &a, int i);
  friend bool operator>(const ieee_floatt &a, int i);
  friend bool operator<(const ieee_floatt &a, int i);
  friend bool operator>=(const ieee_floatt &a, int i);
  friend bool operator<=(const ieee_floatt &a, int i);

protected:
  void divide_and_round(BigInt &fraction, const BigInt &factor);
  void align();
  void next_representable(bool greater);

  // we store the number unpacked
  bool sign_flag;
  BigInt exponent; // this is unbiased
  BigInt fraction; // this _does_ include the hidden bit
  bool NaN_flag, infinity_flag;

  // number of digits of an integer >=1 in base 10
  static BigInt base10_digits(const BigInt &src);
};

bool operator<(const ieee_floatt &a, const ieee_floatt &b);
bool operator<=(const ieee_floatt &a, const ieee_floatt &b);
bool operator>(const ieee_floatt &a, const ieee_floatt &b);
bool operator>=(const ieee_floatt &a, const ieee_floatt &b);
bool operator==(const ieee_floatt &a, const ieee_floatt &b);
bool operator!=(const ieee_floatt &a, const ieee_floatt &b);
std::ostream &operator<<(std::ostream &, const ieee_floatt &);

#endif
