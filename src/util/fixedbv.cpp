/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/arith_tools.h>
#include <util/fixedbv.h>
#include <util/std_types.h>

fixedbv_spect::fixedbv_spect(const fixedbv_typet &type)
{
  integer_bits = type.get_integer_bits();
  width = type.get_width();
}

fixedbv_spect::fixedbv_spect(const fixedbv_type2tc &type)
{
  integer_bits = type->integer_bits;
  width = type->get_width();
}

const fixedbv_type2tc fixedbv_spect::get_type() const
{
  return fixedbv_type2tc(width, integer_bits);
}

fixedbvt::fixedbvt() : v(0)
{
}

fixedbvt::fixedbvt(const fixedbv_spect &s) : spec(s), v(0)
{
}

fixedbvt::fixedbvt(const constant_exprt &expr)
{
  from_expr(expr);
}

void fixedbvt::from_expr(const constant_exprt &expr)
{
  spec = to_fixedbv_type(expr.type());
  v = binary2integer(id2string(expr.get_value()), true);
}

void fixedbvt::from_integer(const BigInt &i)
{
  v = i * power(2, spec.get_fraction_bits());
}

BigInt fixedbvt::to_integer() const
{
  // this rounds to zero, i.e., we just divide
  return v / power(2, spec.get_fraction_bits());
}

constant_exprt fixedbvt::to_expr() const
{
  fixedbv_typet type;
  type.set_width(spec.width);
  type.set_integer_bits(spec.integer_bits);
  constant_exprt expr(type);
  assert(spec.width != 0);
  expr.set_value(integer2binary(v, spec.width));
  return expr;
}

void fixedbvt::round(const fixedbv_spect &dest_spec)
{
  unsigned old_fraction_bits = spec.width - spec.integer_bits;
  unsigned new_fraction_bits = dest_spec.width - dest_spec.integer_bits;

  BigInt result = v;

  if(new_fraction_bits > old_fraction_bits)
    result = v * power(2, new_fraction_bits - old_fraction_bits);
  else if(new_fraction_bits < old_fraction_bits)
  {
    // may need to round
    BigInt p = power(2, old_fraction_bits - new_fraction_bits);
    BigInt div = v / p;
    BigInt rem = v % p;
    if(rem < 0)
      rem = -rem;

    if(rem * 2 >= p)
    {
      if(v < 0)
        --div;
      else
        ++div;
    }

    result = div;
  }

  unsigned old_integer_bits = spec.integer_bits;
  unsigned new_integer_bits = dest_spec.integer_bits;

  if(old_integer_bits > new_integer_bits)
  {
    // Need to cut off some higher bits.
    fixedbvt tmp;
    tmp.spec = dest_spec;

    // Make a number that's 2^integer_bits
    BigInt aval(2);
    aval = power(aval, new_integer_bits);
    tmp.from_integer(aval);

    // Now modulus that up.
    result = result % tmp.v;
  }

  // Increasing integer bits requires no additional changes to representation.

  v = result;
  spec = dest_spec;
}

void fixedbvt::negate()
{
  v = -v;
}

fixedbvt &fixedbvt::operator*=(const fixedbvt &o)
{
  v *= o.v;

  fixedbv_spect old_spec = spec;

  spec.width += o.spec.width;
  spec.integer_bits += o.spec.integer_bits;

  round(old_spec);

  return *this;
}

fixedbvt &fixedbvt::operator/=(const fixedbvt &o)
{
  v *= power(2, o.spec.get_fraction_bits());
  v /= o.v;

  return *this;
}

bool fixedbvt::operator==(int i) const
{
  return v == power(2, spec.get_fraction_bits()) * i;
}

std::string fixedbvt::format(const format_spect &format_spec) const
{
  std::string dest;
  unsigned fraction_bits = spec.get_fraction_bits();

  BigInt int_value = v;
  BigInt factor = power(2, fraction_bits); //BigInt(1)<<fraction_bits;

  if(int_value.is_negative())
  {
    dest += '-';
    int_value.negate();
  }

  std::string base_10_string =
    integer2string(int_value * power(10, fraction_bits) / factor);

  while(base_10_string.size() <= fraction_bits)
    base_10_string = "0" + base_10_string;

  std::string integer_part =
    std::string(base_10_string, 0, base_10_string.size() - fraction_bits);

  std::string fraction_part =
    std::string(base_10_string, base_10_string.size() - fraction_bits);

  dest += integer_part;

  // strip trailing zeros
  while(!fraction_part.empty() &&
        fraction_part[fraction_part.size() - 1] == '0')
    fraction_part.resize(fraction_part.size() - 1);

  if(!fraction_part.empty())
    dest += "." + fraction_part;

  while(dest.size() < format_spec.min_width)
    dest = " " + dest;

  return dest;
}

fixedbvt &fixedbvt::operator+=(const fixedbvt &o)
{
  v += o.v;

  // No need to change the spec.
  round(spec);

  return *this;
}

fixedbvt &fixedbvt::operator-=(const fixedbvt &o)
{
  v -= o.v;

  // No need to change the spec.
  round(spec);

  return *this;
}

fixedbvt &fixedbvt::operator-()
{
  this->negate();
  return (*this);
}

bool operator>(const fixedbvt &a, int i)
{
  fixedbvt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a > other;
}

bool operator<(const fixedbvt &a, int i)
{
  fixedbvt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a < other;
}

bool operator>=(const fixedbvt &a, int i)
{
  fixedbvt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a >= other;
}

bool operator<=(const fixedbvt &a, int i)
{
  fixedbvt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a <= other;
}
