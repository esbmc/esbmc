#include <util/arith/arith_tools.h>
#include <util/arith/fixedbv.h>
#include <util/irep/std_types.h>

fixedbv_spect::fixedbv_spect(const fixedbv_typet &type)
{
  integer_bits = type.get_integer_bits();
  width = type.get_width();
  is_signed = type.get("#esbmc_unsigned") != "1";
  is_saturating = type.get("#esbmc_saturating") == "1";
}

fixedbv_spect::fixedbv_spect(const fixedbv_type2t &type)
{
  integer_bits = type.integer_bits;
  width = type.get_width();
  is_signed = type.is_signed;
  is_saturating = type.is_saturating;
}

type2tc fixedbv_spect::get_type() const
{
  return fixedbv_type2tc(width, integer_bits, is_signed, is_saturating);
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
  v = binary2integer(id2string(expr.get_value()), spec.is_signed);
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
  if (!spec.is_signed)
    type.set("#esbmc_unsigned", "1");
  if (spec.is_saturating)
    type.set("#esbmc_saturating", "1");
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

  if (new_fraction_bits > old_fraction_bits)
    result = v * power(2, new_fraction_bits - old_fraction_bits);
  else if (new_fraction_bits < old_fraction_bits)
  {
    // Narrowing rounds down (floor), matching Clang's fixed-point
    // multiplication and format-conversion semantics (llvm.smul.fix et al.,
    // pinned by the execution oracle). BigInt division truncates toward
    // zero, so adjust negative inexact quotients.
    BigInt p = power(2, old_fraction_bits - new_fraction_bits);
    BigInt div = v / p;
    BigInt rem = v % p;
    if (!rem.is_zero() && v.is_negative())
      --div;

    result = div;
  }

  if (spec.integer_bits > dest_spec.integer_bits)
    // Cut off the high bits, keeping the destination's two's-complement
    // reading. BigInt's % truncates toward zero rather than wrapping, so it
    // could leave a value outside the destination's range -- a constant that
    // disagrees with the encoder's mkBVExtract and flips verdicts under
    // --no-simplify.
    result = binary2integer(
      integer2binary(result, dest_spec.width), dest_spec.is_signed);

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

  this->round(old_spec);

  return *this;
}

fixedbvt &fixedbvt::operator/=(const fixedbvt &o)
{
  // Division rounds down (floor), matching Clang / llvm.sdiv.fix (pinned by
  // the execution oracle). BigInt division truncates toward zero, so adjust
  // inexact quotients whose remainder sign differs from the divisor's.
  v *= power(2, o.spec.get_fraction_bits());
  BigInt rem = v % o.v;
  v /= o.v;
  if (!rem.is_zero() && rem.is_negative() != o.v.is_negative())
    --v;

  return *this;
}

fixedbvt &fixedbvt::operator%=(const fixedbvt &y)
{
  fixedbvt z = *this;
  z /= y;
  z.from_integer(z.to_integer());
  z *= y;
  // ensure z has the same sign as *this
  if (v.is_negative() != z.v.is_negative())
    z.v.negate();
  *this -= z;
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

  if (int_value.is_negative())
  {
    dest += '-';
    int_value.negate();
  }

  std::string base_10_string =
    integer2string(int_value * power(10, fraction_bits) / factor);

  while (base_10_string.size() <= fraction_bits)
    base_10_string = "0" + base_10_string;

  std::string integer_part =
    std::string(base_10_string, 0, base_10_string.size() - fraction_bits);

  std::string fraction_part =
    std::string(base_10_string, base_10_string.size() - fraction_bits);

  dest += integer_part;

  // strip trailing zeros
  while (!fraction_part.empty() &&
         fraction_part[fraction_part.size() - 1] == '0')
    fraction_part.resize(fraction_part.size() - 1);

  if (!fraction_part.empty())
    dest += "." + fraction_part;

  while (dest.size() < format_spec.min_width)
    dest = " " + dest;

  return dest;
}

fixedbvt &fixedbvt::operator+=(const fixedbvt &o)
{
  v += o.v;

  // No need to change the spec.
  this->round(spec);

  return *this;
}

fixedbvt &fixedbvt::operator-=(const fixedbvt &o)
{
  v -= o.v;

  // No need to change the spec.
  this->round(spec);

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
