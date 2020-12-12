/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <limits>
#include <util/arith_tools.h>
#include <util/ieee_float.h>
#include <util/std_types.h>

BigInt ieee_float_spect::bias() const
{
  return power(2, e - 1) - 1;
}

floatbv_typet ieee_float_spect::to_type() const
{
  floatbv_typet result;
  result.set_f(f);
  result.set_width(width());
  return result;
}

BigInt ieee_float_spect::max_exponent() const
{
  return power(2, e) - 1;
}

BigInt ieee_float_spect::max_fraction() const
{
  return power(2, f) - 1;
}

ieee_float_spect::ieee_float_spect(const floatbv_typet &type)
{
  unsigned width = type.get_width();
  f = type.get_f();
  assert(f != 0);
  assert(f < width);
  e = width - f - 1;
}

ieee_float_spect::ieee_float_spect(const floatbv_type2tc &type)
{
  unsigned width = type->get_width();
  f = type->fraction;
  assert(f != 0);
  assert(f < width);
  e = width - f - 1;
}

const floatbv_type2tc ieee_float_spect::get_type() const
{
  return floatbv_type2tc(f, e);
}

ieee_floatt::ieee_floatt()
  : rounding_mode(ROUND_TO_EVEN),
    sign_flag(false),
    exponent(0),
    fraction(0),
    NaN_flag(false),
    infinity_flag(false)
{
}

ieee_floatt::ieee_floatt(const ieee_float_spect &s)
  : rounding_mode(ROUND_TO_EVEN),
    spec(s),
    sign_flag(false),
    exponent(0),
    fraction(0),
    NaN_flag(false),
    infinity_flag(false)
{
}

ieee_floatt::ieee_floatt(const constant_exprt &expr)
  : rounding_mode(ROUND_TO_EVEN)
{
  from_expr(expr);
}

void ieee_floatt::print(std::ostream &out) const
{
  out << to_ansi_c_string();
}

std::string ieee_floatt::format(const format_spect &format_spec) const
{
  std::string result;

  switch(format_spec.style)
  {
  case format_spect::DECIMAL:
    result += to_string_decimal(format_spec.precision);
    break;

  case format_spect::SCIENTIFIC:
    result += to_string_scientific(format_spec.precision);
    break;

  case format_spect::AUTOMATIC:
  {
    // special cases
    BigInt _exponent, _fraction;
    extract_base10(_fraction, _exponent);

    if(_exponent >= 0)
    {
      if(base10_digits(_fraction) + _exponent >= format_spec.precision)
        result += to_string_scientific(format_spec.precision);
      else
        result += to_string_decimal(format_spec.precision);
    }
    else // _exponent<0
    {
      if(true) //base10_digits(fraction)+_exponent<-4)
        result += to_string_scientific(format_spec.precision);
      else
        result += to_string_decimal(format_spec.precision);
    }
  }
  break;
  }

  while(result.size() < format_spec.min_width)
    result = " " + result;

  return result;
}

BigInt ieee_floatt::base10_digits(const BigInt &src)
{
  BigInt tmp = src;
  assert(tmp >= 0);
  BigInt result = 0;
  while(tmp != 0)
  {
    ++result;
    tmp /= 10;
  }
  return result;
}

std::string ieee_floatt::to_string_decimal(unsigned precision) const
{
  std::string result;

  if(sign_flag)
    result += '-';

  if((NaN_flag || infinity_flag) && !sign_flag)
    result += '+';

  // special cases
  if(NaN_flag)
    result += "NaN";
  else if(infinity_flag)
    result += "inf";
  else if(is_zero())
  {
    result += '0';

    // add zeros, if needed
    if(precision > 0)
    {
      result += '.';
      for(unsigned i = 0; i < precision; i++)
        result += '0';
    }
  }
  else
  {
    BigInt _exponent, _fraction;
    extract_base2(_fraction, _exponent);

    // convert to base 10
    if(_exponent >= 0)
    {
      result += integer2string(_fraction * power(2, _exponent));
      if(precision > 0)
      {
        result += '.';
        for(unsigned i = 0; i < precision; i++)
          result += '0';
      }
    }
    else
    {
      BigInt position = -_exponent;

      // 10/2=5 -- this makes it base 10
      _fraction *= power(5, position);

      // apply rounding
      if(position > precision)
      {
        BigInt r = power(10, position - precision);
        BigInt remainder = _fraction % r;
        _fraction /= r;
        // not sure if this is the right kind of rounding here
        if(remainder >= r / 2)
          ++_fraction;
        position = precision;
      }

      std::string tmp = integer2string(_fraction);

      // pad with zeros, if needed
      while(BigInt(tmp.size()) <= position)
        tmp = "0" + tmp;

      std::size_t dot = tmp.size() - position.to_uint64();
      result += std::string(tmp, 0, dot) + '.';
      result += std::string(tmp, dot, std::string::npos);

      for(BigInt i = position; i < precision; ++i)
        result += '0';
    }
  }

  return result;
}

std::string ieee_floatt::to_string_scientific(unsigned precision) const
{
  std::string result;

  if(sign_flag)
    result += '-';

  if((NaN_flag || infinity_flag) && !sign_flag)
    result += '+';

  // special cases
  if(NaN_flag)
    result += "NaN";
  else if(infinity_flag)
    result += "inf";
  else if(is_zero())
  {
    result += '0';

    // add zeros, if needed
    if(precision > 0)
    {
      result += '.';
      for(unsigned i = 0; i < precision; i++)
        result += '0';
    }

    result += "e0";
  }
  else
  {
    BigInt _exponent, _fraction;
    extract_base10(_fraction, _exponent);

    // C99 appears to say that conversion to decimal should
    // use the currently selected IEEE rounding mode.
    if(base10_digits(_fraction) > precision + 1)
    {
      // re-align
      BigInt distance = base10_digits(_fraction) - (precision + 1);
      BigInt p = power(10, distance);
      BigInt remainder = _fraction % p;
      _fraction /= p;
      _exponent += distance;

      if(remainder == p / 2)
      {
        // need to do rounding mode here
        ++_fraction;
      }
      else if(remainder > p / 2)
        ++_fraction;
    }

    std::string decimals = integer2string(_fraction);

    assert(!decimals.empty());

    // First add top digit to result.
    result += decimals[0];

    // Now add dot and further zeros, if needed.
    if(precision > 0)
    {
      result += '.';

      while(decimals.size() < precision + 1)
        decimals += '0';

      result += decimals.substr(1, precision);
    }

    // add exponent
    result += 'e';

    std::string exponent_str =
      integer2string(base10_digits(_fraction) + _exponent - 1);

    if(exponent_str.size() > 0 && exponent_str[0] != '-')
      result += '+';

    result += exponent_str;
  }

  return result;
}

void ieee_floatt::unpack(const BigInt &i)
{
  assert(spec.f != 0);
  assert(spec.e != 0);

  {
    BigInt tmp = i;

    // split this apart
    BigInt pf = power(2, spec.f);
    fraction = tmp % pf;
    tmp /= pf;

    BigInt pe = power(2, spec.e);
    exponent = tmp % pe;
    tmp /= pe;

    sign_flag = (tmp != 0);
  }

  // NaN?
  if(exponent == spec.max_exponent() && fraction != 0)
  {
    make_NaN();
  }
  else if(exponent == spec.max_exponent() && fraction == 0) // Infinity
  {
    NaN_flag = false;
    infinity_flag = true;
  }
  else if(exponent == 0 && fraction == 0) // zero
  {
    NaN_flag = false;
    infinity_flag = false;
  }
  else if(exponent == 0) // denormal?
  {
    NaN_flag = false;
    infinity_flag = false;
    exponent = -spec.bias() + 1; // NOT -spec.bias()!
  }
  else // normal
  {
    NaN_flag = false;
    infinity_flag = false;
    fraction += power(2, spec.f); // hidden bit!
    exponent -= spec.bias();      // un-bias
  }
}

bool ieee_floatt::is_normal() const
{
  return fraction >= power(2, spec.f);
}

BigInt ieee_floatt::pack() const
{
  BigInt result = 0;

  // sign bit
  if(sign_flag)
    result += power(2, spec.e + spec.f);

  if(NaN_flag)
  {
    result += power(2, spec.f) * spec.max_exponent();
    result += 1;
  }
  else if(infinity_flag)
  {
    result += power(2, spec.f) * spec.max_exponent();
  }
  else if(fraction == 0 && exponent == 0)
  {
  }
  else if(is_normal()) // normal?
  {
    // fraction -- need to hide hidden bit
    result += fraction - power(2, spec.f); // hidden bit

    // exponent -- bias!
    result += power(2, spec.f) * (exponent + spec.bias());
  }
  else // denormal
  {
    result += fraction; // denormal -- no hidden bit
    // the exponent is zero
  }

  return result;
}

void ieee_floatt::extract_base2(BigInt &_fraction, BigInt &_exponent) const
{
  if(is_zero() || is_NaN() || is_infinity())
  {
    _fraction = _exponent = 0;
    return;
  }
  _exponent = exponent;
  _fraction = fraction;

  // adjust exponent
  _exponent -= spec.f;

  // try to integer-ize
  while((_fraction % 2) == 0)
  {
    _fraction /= 2;
    ++_exponent;
  }
}

void ieee_floatt::extract_base10(BigInt &_fraction, BigInt &_exponent) const
{
  if(is_zero() || is_NaN() || is_infinity())
  {
    _fraction = _exponent = 0;
    return;
  }

  _exponent = exponent;
  _fraction = fraction;

  // adjust exponent
  _exponent -= spec.f;

  // now make it base 10
  if(_exponent >= 0)
  {
    _fraction *= power(2, _exponent);
    _exponent = 0;
  }
  else // _exponent<0
  {
    // 10/2=5 -- this makes it base 10
    _fraction *= power(5, -_exponent);
  }

  // try to re-normalize
  while((_fraction % 10) == 0)
  {
    _fraction /= 10;
    ++_exponent;
  }
}

void ieee_floatt::build(const BigInt &_fraction, const BigInt &_exponent)
{
  sign_flag = _fraction < 0;
  fraction = _fraction;
  if(sign_flag)
    fraction = -fraction;
  exponent = _exponent;
  exponent += spec.f;
  align();
}

void ieee_floatt::from_base10(const BigInt &_fraction, const BigInt &_exponent)
{
  NaN_flag = infinity_flag = false;
  sign_flag = _fraction < 0;
  fraction = _fraction;
  if(sign_flag)
    fraction = -fraction;
  exponent = spec.f;
  exponent += _exponent;

  if(_exponent < 0)
  {
    // bring to max. precision
    BigInt e_power = power(2, spec.e);
    fraction *= power(2, e_power);
    exponent -= e_power;
    fraction /= power(5, -_exponent);
  }
  else if(_exponent > 0)
  {
    // fix base
    fraction *= power(5, _exponent);
  }

  align();
}

void ieee_floatt::from_integer(const BigInt &i)
{
  NaN_flag = infinity_flag = sign_flag = false;
  exponent = spec.f;
  fraction = i;
  align();
}

void ieee_floatt::align()
{
  // NaN?
  if(NaN_flag)
  {
    fraction = 0;
    exponent = 0;
    sign_flag = false;
    return;
  }

  // do sign
  if(fraction < 0)
  {
    sign_flag = !sign_flag;
    fraction = -fraction;
  }

  // zero?
  if(fraction == 0)
  {
    exponent = 0;
    return;
  }

  // 'usual case'

  BigInt f_power = power(2, spec.f);
  BigInt f_power_next = power(2, spec.f + 1);

  unsigned lowPower2 = fraction.floorPow2();
  BigInt exponent_offset = 0;

  if(lowPower2 < spec.f) // too small
  {
    exponent_offset -= (spec.f - lowPower2);

    assert(fraction * power(2, (spec.f - lowPower2)) >= f_power);
    assert(fraction * power(2, (spec.f - lowPower2)) < f_power_next);
  }
  else if(lowPower2 > spec.f) // too large
  {
    exponent_offset += (lowPower2 - spec.f);

    assert(fraction / power(2, (lowPower2 - spec.f)) >= f_power);
    assert(fraction / power(2, (lowPower2 - spec.f)) < f_power_next);
  }

  BigInt biased_exponent = exponent + exponent_offset + spec.bias();

  // exponent too large (infinity)?
  if(biased_exponent >= spec.max_exponent())
  {
    // we need to consider the rounding mode here
    switch(rounding_mode)
    {
    case UNKNOWN:
    case NONDETERMINISTIC:
    case ROUND_TO_EVEN:
      infinity_flag = true;
      break;

    case ROUND_TO_MINUS_INF:
      // the result of the rounding is never larger than the argument
      if(sign_flag)
        infinity_flag = true;
      else
        make_fltmax();
      break;

    case ROUND_TO_PLUS_INF:
      // the result of the rounding is never smaller than the argument
      if(sign_flag)
      {
        make_fltmax();
        sign_flag = true; // restore sign
      }
      else
        infinity_flag = true;
      break;

    case ROUND_TO_ZERO:
      if(sign_flag)
      {
        make_fltmax();
        sign_flag = true; // restore sign
      }
      else
        make_fltmax(); // positive
      break;
    default:
      assert(false);
    }

    return; // done
  }
  if(biased_exponent <= 0) // exponent too small?
  {
    // produce a denormal (or zero)
    BigInt new_exponent = BigInt(1) - spec.bias();
    exponent_offset = new_exponent - exponent;
  }

  exponent += exponent_offset;

  if(exponent_offset > 0)
  {
    divide_and_round(fraction, power(2, exponent_offset));

    // rounding might make the fraction too big!
    if(fraction == f_power_next)
    {
      fraction = f_power;
      ++exponent;
    }
  }
  else if(exponent_offset < 0)
    fraction *= power(2, -exponent_offset);

  if(fraction == 0)
    exponent = 0;

  // Update flags
  // if the exponent is all 1, than this is either an infinity or a NaN
  if(exponent == (spec.bias() + 1))
  {
    // Infinity if fraction is all 0
    infinity_flag = (fraction == (spec.max_fraction() + 1));

    // NaN if fraction is anything, except all 0
    NaN_flag = !(fraction == (spec.max_fraction() + 1));
  }
}

void ieee_floatt::divide_and_round(BigInt &fraction, const BigInt &factor)
{
  BigInt remainder = fraction % factor;
  fraction /= factor;

  if(remainder != 0)
  {
    switch(rounding_mode)
    {
    case ROUND_TO_EVEN:
    {
      BigInt factor_middle = factor / 2;
      if(remainder < factor_middle)
      {
        // crop
      }
      else if(remainder > factor_middle)
      {
        ++fraction;
      }
      else // exactly in the middle -- go to even
      {
        if((fraction % 2) != 0)
          ++fraction;
      }
    }
    break;

    case ROUND_TO_ZERO:
      // this means just crop
      break;

    case ROUND_TO_MINUS_INF:
      if(sign_flag)
        ++fraction;
      break;

    case ROUND_TO_PLUS_INF:
      if(!sign_flag)
        ++fraction;
      break;

    default:
      assert(false);
    }
  }
}

constant_exprt ieee_floatt::to_expr() const
{
  constant_exprt result(spec.to_type());
  result.set_value(integer2binary(pack(), spec.width()));
  return result;
}

ieee_floatt &ieee_floatt::operator/=(const ieee_floatt &other)
{
  assert(other.spec.f == spec.f);

  // NaN/x = NaN
  if(NaN_flag)
    return *this;

  // x/Nan = NaN
  if(other.NaN_flag)
  {
    make_NaN();
    return *this;
  }

  // 0/0 = NaN
  if(is_zero() && other.is_zero())
  {
    make_NaN();
    return *this;
  }

  // x/0 = +-inf
  if(other.is_zero())
  {
    infinity_flag = true;
    if(other.sign_flag)
      negate();
    return *this;
  }

  // x/inf = NaN
  if(other.infinity_flag)
  {
    if(infinity_flag)
    {
      make_NaN();
      return *this;
    }

    bool old_sign = sign_flag;
    make_zero();
    sign_flag = old_sign;

    if(other.sign_flag)
      negate();

    return *this;
  } // inf/x = inf
  if(infinity_flag)
  {
    if(other.sign_flag)
      negate();
    return *this;
  }

  exponent -= other.exponent;
  fraction *= power(2, spec.f);

  // to account for error
  fraction *= power(2, spec.f);
  exponent -= spec.f;

  fraction /= other.fraction;

  if(other.sign_flag)
    negate();

  align();

  return *this;
}

ieee_floatt &ieee_floatt::operator*=(const ieee_floatt &other)
{
  assert(other.spec.f == spec.f);

  if(other.NaN_flag)
    make_NaN();
  if(NaN_flag)
    return *this;

  if(infinity_flag || other.infinity_flag)
  {
    if(is_zero() || other.is_zero())
    {
      // special case Inf * 0 is NaN
      make_NaN();
      return *this;
    }

    if(other.sign_flag)
      negate();
    infinity_flag = true;
    return *this;
  }

  exponent += other.exponent;
  exponent -= spec.f;
  fraction *= other.fraction;

  if(other.sign_flag)
    negate();

  align();

  return *this;
}

ieee_floatt &ieee_floatt::operator+=(const ieee_floatt &other)
{
  ieee_floatt _other = other;

  assert(_other.spec == spec);

  if(other.NaN_flag)
    make_NaN();
  if(NaN_flag)
    return *this;

  if(infinity_flag && other.infinity_flag)
  {
    if(sign_flag == other.sign_flag)
      return *this;
    make_NaN();
    return *this;
  }
  if(infinity_flag)
    return *this;
  else if(other.infinity_flag)
  {
    infinity_flag = true;
    sign_flag = other.sign_flag;
    return *this;
  }

  // 0 + 0 needs special treatment for the signs
  if(is_zero() && other.is_zero())
  {
    if(get_sign() == other.get_sign())
      return *this;

    if(rounding_mode == ROUND_TO_MINUS_INF)
    {
      set_sign(true);
      return *this;
    }

    set_sign(false);
    return *this;
  }

  // get smaller exponent
  if(_other.exponent < exponent)
  {
    fraction *= power(2, exponent - _other.exponent);
    exponent = _other.exponent;
  }
  else if(exponent < _other.exponent)
  {
    _other.fraction *= power(2, _other.exponent - exponent);
    _other.exponent = exponent;
  }

  assert(exponent == _other.exponent);

  if(sign_flag)
    fraction.negate();
  if(_other.sign_flag)
    _other.fraction.negate();

  fraction += _other.fraction;

  // on zero, retain original sign
  if(fraction == 0)
  {
    if(rounding_mode == ROUND_TO_MINUS_INF)
      sign_flag = true;
    else
      sign_flag = false;
  }
  else // fraction!=0
  {
    sign_flag = (fraction < 0);
    if(sign_flag)
      fraction.negate();
  }

  align();

  return *this;
}

ieee_floatt &ieee_floatt::operator-=(const ieee_floatt &other)
{
  ieee_floatt _other = other;
  _other.sign_flag = !_other.sign_flag;
  return (*this) += _other;
}

ieee_floatt &ieee_floatt::operator-()
{
  this->negate();
  return (*this);
}

bool operator<(const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.NaN_flag || b.NaN_flag)
    return false;

  // check zero
  if(a.is_zero() && b.is_zero())
    return false;

  // check sign
  if(a.is_zero())
    return !b.sign_flag;
  if(b.is_zero())
    return a.sign_flag;

  // check sign
  if(a.sign_flag != b.sign_flag)
    return a.sign_flag;

  // handle infinity
  if(a.infinity_flag)
  {
    if(b.infinity_flag)
      return false;

    return a.sign_flag;
  }
  else if(b.infinity_flag)
    return !a.sign_flag;

  // check exponent
  if(a.exponent != b.exponent)
  {
    if(a.sign_flag) // both negative
      return a.exponent > b.exponent;

    return a.exponent < b.exponent;
  }

  // check significand
  if(a.sign_flag) // both negative
    return a.fraction > b.fraction;

  return a.fraction < b.fraction;
}

bool operator<=(const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.NaN_flag || b.NaN_flag)
    return false;

  // check zero
  if(a.is_zero() && b.is_zero())
    return true;

  if(a.infinity_flag && b.infinity_flag && a.sign_flag == b.sign_flag)
    return true;

  if(
    !a.infinity_flag && !b.infinity_flag && a.sign_flag == b.sign_flag &&
    a.exponent == b.exponent && a.fraction == b.fraction)
    return true;

  return a < b;
}

bool operator>(const ieee_floatt &a, const ieee_floatt &b)
{
  return b < a;
}

bool operator>=(const ieee_floatt &a, const ieee_floatt &b)
{
  return b <= a;
}

bool operator==(const ieee_floatt &a, const ieee_floatt &b)
{
  // If one is NaN, it's always false
  if(a.NaN_flag || b.NaN_flag)
    return false;

  // Compare infs
  if(a.infinity_flag && b.infinity_flag && (a.sign_flag == b.sign_flag))
    return true;

  // If one is Nan, it's always false
  if(a.infinity_flag || b.infinity_flag)
    return false;

  // Zeros are always equal, despite their sign
  if(a.is_zero() && b.is_zero())
    return true;

  assert(a.spec == b.spec);

  return a.exponent == b.exponent && a.fraction == b.fraction &&
         a.sign_flag == b.sign_flag;
}

bool operator==(const ieee_floatt &a, int i)
{
  ieee_floatt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a == other;
}

bool operator>(const ieee_floatt &a, int i)
{
  ieee_floatt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a > other;
}

bool operator<(const ieee_floatt &a, int i)
{
  ieee_floatt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a < other;
}

bool operator>=(const ieee_floatt &a, int i)
{
  ieee_floatt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a >= other;
}

bool operator<=(const ieee_floatt &a, int i)
{
  ieee_floatt other;
  other.spec = a.spec;
  other.from_integer(i);
  return a <= other;
}

bool operator!=(const ieee_floatt &a, const ieee_floatt &b)
{
  return !(a == b);
}

void ieee_floatt::change_spec(const ieee_float_spect &dest_spec)
{
  BigInt _exponent = exponent - spec.f;
  BigInt _fraction = fraction;

  if(sign_flag)
    _fraction.negate();

  spec = dest_spec;
  if(_fraction == 0)
  {
    // We have a zero. It stays a zero.
    // Don't call build to preserve sign.
  }
  else
    build(_fraction, _exponent);
}

void ieee_floatt::from_expr(const constant_exprt &expr)
{
  spec = to_floatbv_type(expr.type());
  unpack(binary2integer(id2string(expr.get_value()), false));
}

BigInt ieee_floatt::to_integer() const
{
  if(NaN_flag || infinity_flag || is_zero())
    return 0;

  BigInt result = fraction;

  BigInt new_exponent = exponent - spec.f;

  // if the exponent is negative, divide
  if(new_exponent < 0)
    result /= power(2, -new_exponent);
  else
    result *= power(2, new_exponent); // otherwise, multiply

  if(sign_flag)
    result.negate();

  return result;
}

void ieee_floatt::from_double(const double d)
{
  spec.f = 52;
  spec.e = 11;
  assert(spec.width() == 64);

  union
  {
    double d;
    long long unsigned int i;
  } u;

  u.d = d;

  unpack(u.i);
}

void ieee_floatt::from_float(const float f)
{
  spec.f = 23;
  spec.e = 8;
  assert(spec.width() == 32);

  union
  {
    float f;
    unsigned int i;
  } u;

  u.f = f;

  unpack(u.i);
}

void ieee_floatt::make_NaN()
{
  NaN_flag = true;
  //sign=false;
  exponent = 0;
  fraction = 0;
  infinity_flag = false;
}

void ieee_floatt::make_fltmax()
{
  BigInt bit_pattern = power(2, spec.e + spec.f) - 1 - power(2, spec.f);
  unpack(bit_pattern);
}

void ieee_floatt::make_fltmin()
{
  unpack(power(2, spec.f));
}

void ieee_floatt::make_plus_infinity()
{
  NaN_flag = false;
  sign_flag = false;
  exponent = 0;
  fraction = 0;
  infinity_flag = true;
}

void ieee_floatt::make_minus_infinity()
{
  make_plus_infinity();
  sign_flag = true;
}

bool ieee_floatt::is_double() const
{
  return spec.f == 52 && spec.e == 11;
}

bool ieee_floatt::is_float() const
{
  return spec.f == 23 && spec.e == 8;
}

double ieee_floatt::to_double() const
{
  union
  {
    double f;
    unsigned int i;
  } a;

  if(infinity_flag)
  {
    if(sign_flag)
      return -std::numeric_limits<double>::infinity();

    return std::numeric_limits<double>::infinity();
  }

  if(NaN_flag)
  {
    if(sign_flag)
      return -std::numeric_limits<double>::quiet_NaN();

    return std::numeric_limits<double>::quiet_NaN();
  }

  BigInt i = pack();
  assert(i.is_uint64());

  a.i = i.to_uint64();
  return a.f;
}

float ieee_floatt::to_float() const
{
  if(sizeof(unsigned) != sizeof(float))
  {
    throw "ieee_floatt::to_float not supported on this architecture";
  }

  union
  {
    float f;
    unsigned int i;
  } a;

  if(infinity_flag)
  {
    if(sign_flag)
      return -std::numeric_limits<float>::infinity();

    return std::numeric_limits<float>::infinity();
  }

  if(NaN_flag)
  {
    if(sign_flag)
      return -std::numeric_limits<float>::quiet_NaN();

    return std::numeric_limits<float>::quiet_NaN();
  }

  BigInt i = pack();
  assert(i.is_uint64());

  a.i = i.to_uint64();
  return a.f;
}

void ieee_floatt::next_representable(bool greater)
{
  if(is_NaN())
    return;

  bool old_sign = get_sign();

  if(is_zero())
  {
    unpack(1);
    set_sign(!greater);

    return;
  }

  if(is_infinity())
  {
    if(get_sign() == greater)
    {
      make_fltmax();
      set_sign(old_sign);
    }
    return;
  }

  bool dir;
  if(greater)
  {
    if(get_sign())
      dir = false;
    else
      dir = true;
  }
  else
  {
    if(get_sign())
      dir = true;
    else
      dir = false;
  }

  set_sign(false);

  BigInt old = pack();
  if(dir)
    ++old;
  else
    --old;

  unpack(old);

  // sign change impossible (zero case caught earler)
  set_sign(old_sign);
}
