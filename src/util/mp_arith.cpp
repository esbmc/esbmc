/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <util/arith_tools.h>
#include <util/mp_arith.h>

mp_integer operator>>(const mp_integer &a, const mp_integer &b)
{
  mp_integer power=::power(2, b);
  
  if(a>=0)
    return a/power;
  else
  {
    // arithmetic shift right isn't division for negative numbers!
    // http://en.wikipedia.org/wiki/Arithmetic_shift 

    if((a%power)==0)
      return a/power;
    else
      return a/power-1;
  }
}

mp_integer operator<<(const mp_integer &a, const mp_integer &b)
{
  return a*power(2, b);
}

std::ostream& operator<<(std::ostream& out, const mp_integer &n)
{
  out << integer2string(n);
  return out;
}

const mp_integer string2integer(const std::string &n, unsigned base)
{
  for(unsigned i=0; i<n.size(); i++)
    if(!(isalnum(n[i]) || (n[i]=='-' && i==0)))
      return 0;

  return mp_integer(n.c_str(), base);
}

const std::string integer2binary(const mp_integer &n, std::size_t width)
{
  mp_integer a(n);

  if(width==0) return "";

  bool neg=a.is_negative();

  if(neg)
  {
    a.negate();
    a=a-1;
  }

  std::size_t len = a.digits(2) + 2;
  char *buffer=(char *)malloc(len);
  char *s = a.as_string(buffer, len, 2);

  std::string result(s);

  free(buffer);

  if(result.size()<width)
  {
    std::string fill;
    fill.resize(width-result.size(), '0');
    result=fill+result;
  }
  else if(result.size()>width)
    result=result.substr(result.size()-width, width);

  if(neg)
    for(std::size_t i=0; i<result.size(); i++)
      result[i]=(result[i]=='0')?'1':'0';

  return result;
}

const std::string integer2string(const mp_integer &n, unsigned base)
{
  unsigned len = n.digits(base) + 2;
  char *buffer=(char *)malloc(len);
  char *s = n.as_string(buffer, len, base);

  std::string result(s);

  free(buffer);

  return result;
}

const mp_integer binary2integer(const std::string &n, bool is_signed)
{
  if(n.size()==0) return 0;

  mp_integer result=0;
  mp_integer mask=1;
  mask=mask << (n.size()-1);

  for(unsigned i=0; i<n.size(); i++)
  {
    if(n[i]=='0')
    {
    }
    else if(n[i]=='1')
    {
      if(is_signed && i==0)
        result=-mask;
      else
        result=result+mask;
    }
    else
    {
      std::cerr << "Invalid number fed to binary2integer" << std::endl;
      abort();
    }

    mask=mask>>1;
  }

  return result;
}

std::size_t integer2size_t(const mp_integer &n)
{
  assert(n>=0);
  mp_integer::ullong_t ull=n.to_ulong();
  assert(ull <= std::numeric_limits<std::size_t>::max());
  return (std::size_t)ull;
}

unsigned integer2unsigned(const mp_integer &n)
{
  assert(n>=0);
  mp_integer::ullong_t ull=n.to_ulong();
  assert(ull <= std::numeric_limits<unsigned>::max());
  return (unsigned)ull;
}
