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

typedef BigInt::ullong_t ullong_t;
typedef BigInt::llong_t llong_t;

BigInt operator>>(const BigInt &a, const BigInt &b)
{
  BigInt power = ::power(2, b);

  if(a >= 0)
    return a / power;

  // arithmetic shift right isn't division for negative numbers!
  // http://en.wikipedia.org/wiki/Arithmetic_shift

  if((a % power) == 0)
    return a / power;

  return a / power - 1;
}

BigInt operator<<(const BigInt &a, const BigInt &b)
{
  return a * power(2, b);
}

std::ostream &operator<<(std::ostream &out, const BigInt &n)
{
  out << integer2string(n);
  return out;
}

const BigInt string2integer(const std::string &n, unsigned base)
{
  for(unsigned i = 0; i < n.size(); i++)
    if(!(isalnum(n[i]) || (n[i] == '-' && i == 0)))
      return 0;

  return BigInt(n.c_str(), base);
}

const std::string integer2binary(const BigInt &n, std::size_t width)
{
  BigInt a(n);

  if(width == 0)
    return "";

  bool neg = a.is_negative();

  if(neg)
  {
    a.negate();
    a = a - 1;
  }

  std::size_t len = a.digits(2) + 2;
  char *buffer = (char *)malloc(len);
  char *s = a.as_string(buffer, len, 2);

  std::string result(s);

  free(buffer);

  if(result.size() < width)
  {
    std::string fill;
    fill.resize(width - result.size(), '0');
    result = fill + result;
  }
  else if(result.size() > width)
    result = result.substr(result.size() - width, width);

  if(neg)
    for(char &i : result)
      i = (i == '0') ? '1' : '0';

  return result;
}

const std::string integer2string(const BigInt &n, unsigned base)
{
  unsigned len = n.digits(base) + 2;
  char *buffer = (char *)malloc(len);
  char *s = n.as_string(buffer, len, base);

  std::string result(s);

  free(buffer);

  return result;
}

const BigInt binary2integer(const std::string &n, bool is_signed)
{
  if(n.size() == 0)
    return 0;

  BigInt result = 0;
  BigInt mask = 1;
  mask = mask << (n.size() - 1);

  for(unsigned i = 0; i < n.size(); i++)
  {
    if(n[i] == '0')
    {
    }
    else if(n[i] == '1')
    {
      if(is_signed && i == 0)
        result = -mask;
      else
        result = result + mask;
    }
    else
    {
      std::cerr << "Invalid number fed to binary2integer" << std::endl;
      abort();
    }

    mask = mask >> 1;
  }

  return result;
}

std::size_t integer2size_t(const BigInt &n)
{
  assert(n >= 0);
  BigInt::ullong_t ull = n.to_ulong();
  assert(ull <= std::numeric_limits<std::size_t>::max());
  return (std::size_t)ull;
}

unsigned integer2unsigned(const BigInt &n)
{
  assert(n >= 0);
  BigInt::ullong_t ull = n.to_ulong();
  assert(ull <= std::numeric_limits<unsigned>::max());
  return (unsigned)ull;
}

BigInt bitwise(const BigInt &a, const BigInt &b, std::function<bool(bool, bool)> f)
{
  const auto digits = std::max(a.digits(2), b.digits(2));

  BigInt result = 0;
  BigInt tmp_a = a, tmp_b = b;

  for(std::size_t i =0; i < digits; i++)
  {
    const bool bit_a = tmp_a.is_odd();
    const bool bit_b = tmp_b.is_odd();
    const bool bit_result = f(bit_a, bit_b);
    if(bit_result)
      result += power(2, i);
    tmp_a /=2;
    tmp_b /=2;
  }

  return result;
}



BigInt bitwise_or(const BigInt &a, const BigInt &b)
{
  (!a.is_negative() && !b.is_negative());

  if(a.is_ulong() && b.is_ulong())
    return a.to_ulong() | b.to_ulong();

  return bitwise(a, b, [](bool a , bool b) { return a || b; });
}

BigInt bitwise_and(const BigInt &a, const BigInt &b)
{
  (!a.is_negative() && !b.is_negative());

  if(a.is_ulong() && b.is_ulong())
    return a.to_ulong() & b.to_ulong();

  return bitwise(a, b, [](bool a , bool b) { return a && b; });
}

BigInt bitwise_xor(const BigInt &a, const BigInt &b)
{
  (!a.is_negative() && !b.is_negative());

  if(a.is_ulong() && b.is_ulong())
    return a.to_ulong() ^ b.to_ulong();

  return bitwise(a, b, [](bool a , bool b) { return a != b;});
}


BigInt arith_left_shift(const BigInt &a, const BigInt &b, std::size_t true_size)
{
  (a.is_long() && b.is_ulong());
  (b <= true_size || a == 0);

  ullong_t shift=b.to_ulong();

  llong_t result=a.to_long()<<shift;
  llong_t mask = 
    true_size<(sizeof(llong_t)*8) ?
    (1LL << true_size) -1 :
    -1;
    return result&mask;
}

BigInt arith_right_shift(const BigInt &a, const BigInt &b, std::size_t true_size)
{
  (a.is_long() && b.is_ulong());
  llong_t number=a.to_long();
  ullong_t shift=b.to_ulong();
  (shift <= true_size);

  const llong_t sign = (1LL << (true_size - 1)) & number;
  const llong_t pad = (sign == 0) ? 0 : ~((1LL << (true_size - shift)) - 1);
  llong_t result=(number >> shift)|pad;
  return result;
}

BigInt logic_left_shift(const BigInt &a, const BigInt &b, std::size_t true_size)
{
  (a.is_long() && b.is_ulong());
  (b <= true_size || a == 0);

  ullong_t shift=b.to_ulong();
  llong_t result=a.to_long()<<shift;
  if(true_size<(sizeof(llong_t)*8))
  {
    const llong_t sign = (1LL << (true_size - 1)) & result;
    const llong_t mask = (1LL << true_size) - 1;
    // Sign-fill out-of-range bits:
    if(sign==0)
      result&=mask;
    else
      result|=~mask;
  }
  return result;
}

BigInt logic_right_shift(const BigInt &a, const BigInt &b, std::size_t true_size)
{
  (a.is_long() && b.is_ulong());
  (b <= true_size);

  ullong_t shift = b.to_ulong();
  ullong_t result=((ullong_t)a.to_long()) >> shift;
  return result;
}

BigInt rotate_right(const BigInt &a, const BigInt &b, std::size_t true_size)
{
  (a.is_ulong() && b.is_ulong());
  (b <= true_size);

  ullong_t number=a.to_ulong();
  ullong_t shift=b.to_ulong();

  ullong_t revShift=true_size-shift;
  const ullong_t filter = 1ULL << (true_size - 1);
  ullong_t result=(number >> shift)|((number<<revShift)&filter);
  return result;
}

BigInt rotate_left(const BigInt &a, const BigInt &b, std::size_t true_size)
{
  (a.is_ulong() && b.is_ulong());
  (b <= true_size);

  ullong_t number=a.to_ulong();
  ullong_t shift=b.to_ulong();

  ullong_t revShift=true_size-shift;
  const ullong_t filter = 1ULL << (true_size - 1);
  ullong_t result=((number<<shift)&filter)|((number&filter) >> revShift);
  return result;
}
