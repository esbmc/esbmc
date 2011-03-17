/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <arith_tools.h>
#include <std_types.h>

#include "ieee_float.h"

/*******************************************************************\

Function: ieee_float_spect::bias

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer ieee_float_spect::bias() const
{
  return power(2, e-1)-1;
}

/*******************************************************************\

Function: ieee_float_spect::to_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

floatbv_typet ieee_float_spect::to_type() const
{
  floatbv_typet result;
  result.set_f(f);
  result.set_width(width());
  return result;
}

/*******************************************************************\

Function: ieee_float_spect::max_exponent

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer ieee_float_spect::max_exponent() const
{
  return power(2, e)-1;
}

/*******************************************************************\

Function: ieee_float_spect::max_fraction

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer ieee_float_spect::max_fraction() const
{
  return power(2, f)-1;
}

/*******************************************************************\

Function: ieee_float_spect::from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_float_spect::from_type(const floatbv_typet &type)
{
  unsigned width=type.get_width();
  f=type.get_f();
  assert(f!=0);
  assert(f<width);
  e=width-f-1;
}

/*******************************************************************\

Function: ieee_floatt::print

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::print(std::ostream &out) const
{
  out << to_ansi_c_string();
}

/*******************************************************************\

Function: ieee_floatt::format

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string ieee_floatt::format(const format_spect &format_spec) const
{
  std::string result;

  if(sign) result+="-";
  
  if((NaN || infinity) && !sign) result+="+";

  // special cases
  if(NaN)
    result+="NaN";
  else if(infinity)
    result+="inf";
  else if(is_zero())
    result+="0";
  else
  {
    mp_integer _exponent, _fraction;
    extract(_fraction, _exponent);

    // convert to base 10
    if(_exponent>=0)
    {
      result+=integer2string(_fraction*power(2, _exponent));
    }
    else
    {
      #if 1
      mp_integer position=-_exponent;

      // 10/2=5 -- this makes it base 10
      _fraction*=power(5, position);

      // apply rounding
      if(position>format_spec.precision)
      {
        mp_integer r=power(10, position-format_spec.precision);
        mp_integer remainder=_fraction%r;
        _fraction/=r;
        // not sure if this is the right kind of rounding here
        if(remainder>=r/2) ++_fraction;
        position=format_spec.precision;
      }

      std::string tmp=integer2string(_fraction);

      // pad with zeros, if needed
      while(mp_integer(tmp.size())<=position) tmp="0"+tmp;

      unsigned dot=tmp.size()-integer2long(position);
      result+=std::string(tmp, 0, dot)+'.';
      result+=std::string(tmp, dot, std::string::npos);

      #else

      result+=integer2string(_fraction);

      if(_exponent!=0)
        result+="*2^"+integer2string(_exponent);

      #endif
    }
  }

  while(result.size()<format_spec.min_width)
    result=" "+result;

  return result;
}

/*******************************************************************\

Function: ieee_floatt::unpack

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::unpack(const mp_integer &i)
{
  assert(spec.f!=0);
  assert(spec.e!=0);

  {
    mp_integer tmp=i;

    // split this apart
    mp_integer pf=power(2, spec.f);
    fraction=tmp%pf;
    tmp/=pf;

    mp_integer pe=power(2, spec.e);
    exponent=tmp%pe;
    tmp/=pe;

    sign=(tmp!=0);
  }

  // NaN?
  if(exponent==spec.max_exponent() && fraction!=0)
  {
    make_NaN();
  }
  else if(exponent==spec.max_exponent() && fraction==0) // Infinity
  {
    NaN=false;
    infinity=true;
  }
  else if(exponent==0 && fraction==0) // zero
  {
    NaN=false;
    infinity=false;
  }
  else if(exponent==0) // denormal?
  {
    NaN=false;
    infinity=false;
    exponent=-spec.bias()+1; // NOT -spec.bias()!
  }
  else // normal
  {
    NaN=false;
    infinity=false;
    fraction+=power(2, spec.f); // hidden bit!    
    exponent-=spec.bias(); // un-bias
  }
}

/*******************************************************************\

Function: ieee_floatt::pack

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer ieee_floatt::pack() const
{
  mp_integer result=0;

  // sign bit
  if(sign) result+=power(2, spec.e+spec.f);

  if(NaN)
  {
    result+=power(2, spec.f)*spec.max_exponent();
    result+=1;
  }
  else if(infinity)
  {
    result+=power(2, spec.f)*spec.max_exponent();
  }
  else if(fraction==0 && exponent==0)
  {
  }
  else if(fraction>=power(2, spec.f)) // normal?
  {
    // fraction -- need to hide hidden bit
    result+=fraction-power(2, spec.f); // hidden bit

    // exponent -- bias!
    result+=power(2, spec.f)*(exponent+spec.bias());
  }
  else // denormal
  {
    result+=fraction; // denormal -- no hidden bit
    // the exponent is zero
  }

  return result;
}

/*******************************************************************\

Function: ieee_floatt::extract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::extract(
  mp_integer &_fraction,
  mp_integer &_exponent) const
{
  _exponent=exponent;
  _fraction=fraction;

  // adjust exponent
  _exponent-=spec.f;

  // try to integer-ize
  while((_fraction%2)==0)
  {
    _fraction/=2;
    ++_exponent;
  }
}

/*******************************************************************\

Function: ieee_floatt::build

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::build(
  const mp_integer &_fraction,
  const mp_integer &_exponent)
{
  NaN=infinity=false;
  sign=_fraction<0;
  fraction=_fraction;
  if(sign) fraction=-fraction;
  exponent=_exponent;
  exponent+=spec.f;
  align();
}

/*******************************************************************\

Function: ieee_floatt::from_base10

  Inputs:

 Outputs:

 Purpose: compute f * (10^e)

\*******************************************************************/

void ieee_floatt::from_base10(
  const mp_integer &_fraction,
  const mp_integer &_exponent)
{
  NaN=infinity=false;
  sign=_fraction<0;
  fraction=_fraction;
  if(sign) fraction=-fraction;
  exponent=spec.f;
  exponent+=_exponent;
  
  if(_exponent<0)
  {
    // bring to max. precision
    mp_integer e_power=power(2, spec.e);
    fraction*=power(2, e_power);
    exponent-=e_power;
    fraction/=power(5, -_exponent);
  }
  else if(_exponent>0)
  {
    // fix base
    fraction*=power(5, _exponent);
  }
  
  align();
}

/*******************************************************************\

Function: ieee_floatt::from_integer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::from_integer(const mp_integer &i)
{
  NaN=infinity=sign=false;
  exponent=spec.f;
  fraction=i;
  align();
}

/*******************************************************************\

Function: ieee_floatt::align

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::align()
{
  // NaN?
  if(NaN)
  {
    fraction=exponent=0;
    sign=false;
    return;
  }

  // do sign
  if(fraction<0)
  {
    sign=!sign;
    fraction=-fraction;
  }

  // zero?
  if(fraction==0)
  {
    exponent=0;
    return;
  }

  // 'usual case'

  mp_integer f_power=power(2, spec.f);
  mp_integer f_power_next=power(2, spec.f+1);

  mp_integer exponent_offset=0;

  if(fraction<f_power) // too small?
  {
    mp_integer tmp_fraction=fraction;

    while(tmp_fraction<f_power)
    {
      tmp_fraction*=2;
      --exponent_offset;
    }
  }
  else if(fraction>=f_power_next) // too big?
  {
    mp_integer tmp_fraction=fraction;

    while(tmp_fraction>=f_power_next)
    {
      tmp_fraction/=2;
      ++exponent_offset;
    }
  }

  mp_integer biased_exponent=exponent+exponent_offset+spec.bias();

  // exponent too large (infinity)?
  if(biased_exponent>=spec.max_exponent())
    infinity=true;
  else if(biased_exponent<=0) // exponent too small?
  {
    // produce a denormal (or zero)
    mp_integer new_exponent=mp_integer(1)-spec.bias();
    exponent_offset=new_exponent-exponent;
  }

  exponent+=exponent_offset;

  if(exponent_offset>0)
  {
    divide_and_round(fraction, power(2, exponent_offset));

    // rounding might make the fraction too big!
    if(fraction==f_power_next)
    {
      fraction=f_power;
      ++exponent;
    }
  }
  else if(exponent_offset<0)
    fraction*=power(2, -exponent_offset);

  if(fraction==0)
    exponent=0;
}

/*******************************************************************\

Function: ieee_floatt::divide_and_round

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::divide_and_round(
  mp_integer &fraction,
  const mp_integer &factor)
{
  mp_integer remainder=fraction%factor;
  fraction/=factor;

  if(remainder!=0)
  {
    switch(rounding_mode)
    {
    case ROUND_TO_EVEN:
      {
        mp_integer factor_middle=factor/2;
        if(remainder<factor_middle)
        {
          // crop
        }
        else if(remainder>factor_middle)
        {
          ++fraction;
        }
        else // exactly in the middle -- go to even
        {
          if((fraction%2)!=0)
            ++fraction;
        }
      }
      break;

    case ROUND_TO_ZERO:
      // this means just crop
      break;

    case ROUND_TO_MINUS_INF:
      if(sign)
        ++fraction;
      break;

    case ROUND_TO_PLUS_INF:
      if(!sign)
        ++fraction;
      break;

    default:
      assert(false);
    }
  }
}

/*******************************************************************\

Function: ieee_floatt::to_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt ieee_floatt::to_expr() const
{
  exprt result=exprt("constant", spec.to_type());
  result.set("value", integer2binary(pack(), spec.width()));
  return result;
}

/*******************************************************************\

Function: operator /=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

ieee_floatt &ieee_floatt::operator /= (const ieee_floatt &other)
{
  assert(other.spec.f==spec.f);
  
  // NaN/x = NaN
  if(NaN) return *this;
  
  // x/Nan = NaN
  if(other.NaN) { make_NaN(); return *this; }
  
  // 0/0 = NaN
  if(is_zero() && other.is_zero()) { make_NaN(); return *this; }

  // x/0 = +-inf
  if(other.is_zero())
  {
    infinity=true;
    if(other.sign) negate();
    return *this;
  }
  
  // x/inf = NaN
  if(other.infinity)
  {
    if(infinity) { make_NaN(); return *this; }
    make_zero();
    return *this;
  }

  exponent-=other.exponent;
  fraction*=power(2, spec.f);

  // to account for error
  fraction*=power(2, spec.f);
  exponent-=spec.f;

  fraction/=other.fraction;

  if(other.sign) negate();

  align();

  return *this;
}

/*******************************************************************\

Function: operator *=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

ieee_floatt &ieee_floatt::operator *= (const ieee_floatt &other)
{
  assert(other.spec.f==spec.f);
  
  if(other.NaN) make_NaN();
  if(NaN) return *this;
  
  if(infinity || other.infinity)
  {
    if(is_zero() || other.is_zero())
    {
      // special case Inf * 0 is NaN
      make_NaN();
      return *this;
    }

    if(other.sign) negate();
    infinity=true;
    return *this;
  }

  exponent+=other.exponent;
  exponent-=spec.f;
  fraction*=other.fraction;

  if(other.sign) negate();

  align();

  return *this;
}

/*******************************************************************\

Function: operator +=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

ieee_floatt &ieee_floatt::operator += (const ieee_floatt &other)
{
  ieee_floatt _other=other;

  assert(_other.spec==spec);
  
  if(other.NaN) make_NaN();
  if(NaN) return *this;

  if(infinity && other.infinity)
  {
    if(sign==other.sign) return *this;
    make_NaN();
    return *this;
  }
  else if(infinity)
    return *this;
  else if(other.infinity)
  {
    infinity=true;
    sign=other.sign;
    return *this;
  }
  
  // get smaller exponent
  if(_other.exponent<exponent)
  {
    fraction*=power(2, exponent-_other.exponent);
    exponent=_other.exponent;
  }
  else if(exponent<_other.exponent)
  {
    _other.fraction*=power(2, _other.exponent-exponent);
    _other.exponent=exponent;
  }
  
  assert(exponent==_other.exponent);

  if(sign) fraction.negate();
  if(_other.sign) _other.fraction.negate();
  
  fraction+=_other.fraction;
  
  // on zero, retain original sign
  if(fraction!=0)
  {
    sign=(fraction<0);
    if(sign) fraction.negate();
  }

  align();

  return *this;
}

/*******************************************************************\

Function: operator -=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

ieee_floatt &ieee_floatt::operator -= (const ieee_floatt &other)
{
  ieee_floatt _other=other;
  _other.sign=!_other.sign;
  return (*this)+=_other;
}

/*******************************************************************\

Function: operator <

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator < (const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.NaN || b.NaN) return false;
  
  // check zero
  if(a.is_zero() && b.is_zero())
    return false;

  // check sign
  if(a.sign!=b.sign)
    return a.sign;
    
  // check exponent
  if(a.exponent!=b.exponent)
  {
    if(a.sign) // both negative
      return a.exponent>b.exponent;
    else
      return a.exponent<b.exponent;
  }
  
  // check significand
  if(a.sign) // both negative
    return a.fraction>b.fraction;
  else
    return a.fraction<b.fraction;
}

/*******************************************************************\

Function: operator <=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator <=(const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.NaN || b.NaN) return false;
  
  // check zero
  if(a.is_zero() && b.is_zero())
    return true;

  if(a.sign==b.sign &&
     a.exponent==b.exponent &&
     a.fraction==b.fraction)
    return true;
    
  return a<b;
}

/*******************************************************************\

Function: operator >

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator > (const ieee_floatt &a, const ieee_floatt &b)
{
  return b < a;
}

/*******************************************************************\

Function: operator >=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator >=(const ieee_floatt &a, const ieee_floatt &b)
{
  return b <= a;
}

/*******************************************************************\

Function: operator ==

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator ==(const ieee_floatt &a, const ieee_floatt &b)
{
  // packed equality!
  if(a.NaN && b.NaN) return true;

  if(a.infinity && b.infinity) return true;

  if(a.is_zero() && b.is_zero()) return true;

  return a.exponent==b.exponent &&
         a.fraction==b.fraction &&
         a.sign==b.sign;
}

/*******************************************************************\

Function: ieee_equal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ieee_equal(const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.NaN || b.NaN) return false;
  if(a.is_zero() && b.is_zero()) return true;
  assert(a.spec==b.spec);
  return a==b;
}

/*******************************************************************\

Function: operator ==

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator ==(const ieee_floatt &a, int i)
{
  ieee_floatt other;
  other.spec=a.spec;
  other.from_integer(i);
  return a==other;
}

/*******************************************************************\

Function: operator !=

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator !=(const ieee_floatt &a, const ieee_floatt &b)
{
  return !(a==b);
}

/*******************************************************************\

Function: ieee_not_equal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool ieee_not_equal(const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.NaN || b.NaN) return true; // !!!
  if(a.is_zero() && b.is_zero()) return false;
  assert(a.spec==b.spec);
  return a!=b;
}

/*******************************************************************\

Function: ieee_floatt::change_spec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::change_spec(const ieee_float_spect &dest_spec)
{
  mp_integer _exponent=exponent-spec.f;
  mp_integer _fraction=fraction;
  
  if(sign) _fraction.negate();

  spec=dest_spec;
  build(_fraction, _exponent);
}

/*******************************************************************\

Function: ieee_floatt::from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::from_expr(const exprt &expr)
{
  assert(expr.is_constant());
  spec=to_floatbv_type(expr.type());
  unpack(binary2integer(expr.get_string("value"), false));
}

/*******************************************************************\

Function: ieee_floatt::to_integer

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

mp_integer ieee_floatt::to_integer() const
{
  if(NaN || infinity || is_zero()) return 0;

  mp_integer result=fraction;

  mp_integer new_exponent=exponent-spec.f;
  
  // if the exponent is negative, divide
  if(new_exponent<0)
    result/=power(2, -new_exponent);
  else  
    result*=power(2, new_exponent); // otherwise, multiply

  if(sign)
    result.negate();
    
  return result;
}

/*******************************************************************\

Function: ieee_floatt::from_double

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::from_double(const double d)
{
  spec.f=52;
  spec.e=11;
  assert(spec.width()==64);
  
  union
  {
    double d;
    long long unsigned int i;
  } u;
  
  u.d=d;
  
  unpack(u.i);
}

/*******************************************************************\

Function: ieee_floatt::from_float

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::from_float(const float f)
{
  spec.f=23;
  spec.e=8;
  assert(spec.width()==32);

  union
  {
    float f;
    long unsigned int i;
  } u;
  
  u.f=f;

  unpack((unsigned long long)u.i);
}

/*******************************************************************\

Function: ieee_floatt::make_NaN

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void ieee_floatt::make_NaN()
{
  NaN=true;
  sign=false;
  exponent=fraction=0;
  infinity=false;
}
