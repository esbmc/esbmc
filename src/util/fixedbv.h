/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_FIXEDBV_UTIL_H
#define CPROVER_FIXEDBV_UTIL_H

#include <util/format_spec.h>
#include <util/irep2_type.h>
#include <util/mp_arith.h>
#include <util/std_expr.h>
#include <util/std_types.h>

class fixedbv_spect
{
public:
  unsigned integer_bits, width;

  fixedbv_spect() : integer_bits(0), width(0)
  {
  }

  fixedbv_spect(unsigned _width, unsigned _integer_bits)
    : integer_bits(_integer_bits), width(_width)
  {
    assert(width >= integer_bits);
  }

  fixedbv_spect(const fixedbv_typet &type);
  fixedbv_spect(const fixedbv_type2tc &type);

  inline unsigned get_fraction_bits() const
  {
    return width - integer_bits;
  }

  const fixedbv_type2tc get_type() const;
};

class fixedbvt
{
public:
  fixedbv_spect spec;

  fixedbvt();
  fixedbvt(const fixedbv_spect &s);

  explicit fixedbvt(const constant_exprt &expr);

  void from_integer(const BigInt &i);
  BigInt to_integer() const; // this rounds to zero
  void from_expr(const constant_exprt &expr);
  constant_exprt to_expr() const;
  void round(const fixedbv_spect &dest_spec);

  std::string to_ansi_c_string() const
  {
    return format(format_spect());
  }

  std::string format(const format_spect &format_spec) const;

  bool operator==(int i) const;
  bool is_zero() const
  {
    return v == 0;
  }

  bool is_negative() const
  {
    return v.is_negative();
  }

  bool get_sign() const
  {
    return v.is_negative();
  }

  // Never true
  bool is_NaN() const
  {
    return false;
  }

  // Never true
  bool is_infinity() const
  {
    return false;
  }

  // Always true
  bool is_finite() const
  {
    return true;
  }

  // Always true?
  bool is_normal() const
  {
    return true;
  }

  void negate();

  fixedbvt &operator/=(const fixedbvt &other);
  fixedbvt &operator*=(const fixedbvt &other);
  fixedbvt &operator+=(const fixedbvt &other);
  fixedbvt &operator-=(const fixedbvt &other);

  fixedbvt &operator-();

  friend bool operator<(const fixedbvt &a, const fixedbvt &b)
  {
    return a.v < b.v;
  }
  friend bool operator<=(const fixedbvt &a, const fixedbvt &b)
  {
    return a.v <= b.v;
  }
  friend bool operator>(const fixedbvt &a, const fixedbvt &b)
  {
    return a.v > b.v;
  }
  friend bool operator>=(const fixedbvt &a, const fixedbvt &b)
  {
    return a.v >= b.v;
  }
  friend bool operator==(const fixedbvt &a, const fixedbvt &b)
  {
    return a.v == b.v;
  }
  friend bool operator!=(const fixedbvt &a, const fixedbvt &b)
  {
    return a.v != b.v;
  }

  friend bool operator>(const fixedbvt &a, int i);
  friend bool operator<(const fixedbvt &a, int i);
  friend bool operator>=(const fixedbvt &a, int i);
  friend bool operator<=(const fixedbvt &a, int i);

  const BigInt &get_value() const
  {
    return v;
  }
  void set_value(const BigInt &_v)
  {
    v = _v;
  }

protected:
  // negative values stored as such
  BigInt v;
};

bool operator<(const fixedbvt &a, const fixedbvt &b);
bool operator<=(const fixedbvt &a, const fixedbvt &b);
bool operator>(const fixedbvt &a, const fixedbvt &b);
bool operator>=(const fixedbvt &a, const fixedbvt &b);
bool operator==(const fixedbvt &a, const fixedbvt &b);
bool operator!=(const fixedbvt &a, const fixedbvt &b);

#endif
