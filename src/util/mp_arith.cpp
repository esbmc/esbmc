#include <cassert>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <util/arith_tools.h>
#include <util/mp_arith.h>

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

/// \par parameters: string of '0'-'9' etc. most significant digit first
/// base of number representation
/// \return BigInt
const BigInt string2integer(const std::string &n, unsigned base)
{
  for(std::size_t i = 0; i < n.size(); i++)
    if(!(isalnum(n[i]) || (n[i] == '-' && i == 0)))
      return 0;

  return BigInt(n.c_str(), base);
}

/// \return string of '0'/'1', most significant bit first
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
  std::vector<char> buffer(len);
  char *s = a.as_string(buffer.data(), len, 2);

  std::string result(s);

  if(result.size() < width)
  {
    std::string fill;
    fill.resize(width - result.size(), '0');
    result = fill + result;
  }
  else if(result.size() > width)
    result = result.substr(result.size() - width, width);

  if(neg)
    for(std::size_t i = 0; i < result.size(); i++)
      result[i] = (result[i] == '0') ? '1' : '0';

  return result;
}

const std::string integer2string(const BigInt &n, unsigned base)
{
  unsigned len = n.digits(base) + 2;
  std::vector<char> buffer(len);
  char *s = n.as_string(buffer.data(), len, base);

  std::string result(s);

  return result;
}

/// convert binary string representation to BigInt
/// \par parameters: string of '0'/'1', most significant bit first
/// \return BigInt
const BigInt binary2integer(const std::string &n, bool is_signed)
{
  if(n.empty())
    return 0;

  if(n.size() <= (sizeof(unsigned long) * 8))
  {
    // this is a tuned implementation for short integers

    unsigned long mask = 1;
    mask = mask << (n.size() - 1);
    BigInt top_bit = (n[0] == '1') ? mask : 0;
    if(is_signed)
      top_bit.negate();
    mask >>= 1;
    unsigned long other_bits = 0;

    for(std::string::const_iterator it = ++n.begin(); it != n.end(); ++it)
    {
      if(*it == '1')
        other_bits += mask;
      else if(*it != '0')
        return 0;

      mask >>= 1;
    }

    return top_bit + other_bits;
  }

  if(n.find_first_not_of("01") != std::string::npos)
    return 0;

  if(is_signed && n[0] == '1')
  {
    BigInt result(n.c_str() + 1, 2);
    result -= BigInt(1) << (n.size() - 1);
    return result;
  }
  else
    return BigInt(n.c_str(), 2);
}
