/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdlib.h>
#include <ctype.h>

#include <sstream>

#include "mp_arith.h"

mp_integer operator>>(const mp_integer &a, unsigned int b)
{
  mp_integer result=a;

  for(unsigned i=0; i<b; i++)
    result/=2;

  return result;
}

mp_integer operator<<(const mp_integer &a, unsigned int b)
{
  mp_integer result=a;

  for(unsigned i=0; i<b; i++)
    result*=2;

  return result;
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

const std::string integer2binary(const mp_integer &n, unsigned width)
{
  mp_integer a(n);

  if(width==0) return "";

  bool neg=a.is_negative();

  if(neg)
  {
    a.negate();
    a=a-1;
  }

  unsigned len = a.digits(2) + 2;
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
    for(unsigned i=0; i<result.size(); i++)
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

const mp_integer extract_fraction(const std::string &n, bool is_signed, u_int from, u_int to)
{
  if(n.size()==0) return 0;

  mp_integer result=0;
  mp_integer mask=1;
  mask=mask << (to-1);

  for(unsigned i=from; i<to; i++)
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
      return 0;

    mask=mask>>1;
  }

  return result;
}

unsigned long integer2long(const mp_integer &n)
{
  return n.to_ulong();
}
