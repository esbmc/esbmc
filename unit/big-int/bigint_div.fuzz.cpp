/*******************************************************************
 Module: BigInt division fuzz test

 Fuzz Plan:
   - Differentially test BigInt::div against operator/ and operator%
   - Assert the division identity  x == q*y + r  for arbitrary operands
   - Stresses the single-limb-divisor / large-dividend path that once
     returned quotient 1, remainder 0 (see BigInt::div).
 \*******************************************************************/

#include <cctype>
#include <cassert>
#include <cstring>
#include <string>
#include <big-int/bigint.hh>
#include <stddef.h>

// Parse a signed decimal string into `out`. Returns false when the string is
// empty or contains anything other than an optional leading '-' and digits.
static bool parse_decimal(const std::string &s, BigInt &out)
{
  size_t i = (!s.empty() && s[0] == '-') ? 1 : 0;
  if (i == s.size())
    return false; // empty, or just "-"
  for (size_t j = i; j < s.size(); ++j)
    if (!std::isdigit(static_cast<unsigned char>(s[j])))
      return false;
  out = BigInt(s.c_str(), 10);
  return true;
}

extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size)
{
  // Split the input into two decimal operands at the first '/'.
  const char *slash = static_cast<const char *>(memchr(Data, '/', Size));
  if (slash == nullptr)
    return -1;

  std::string lhs(Data, slash - Data);
  std::string rhs(slash + 1, (Data + Size) - (slash + 1));

  BigInt x, y;
  if (!parse_decimal(lhs, x) || !parse_decimal(rhs, y))
    return -1;
  if (y.is_zero())
    return -1;

  BigInt q, r;
  BigInt::div(x, y, q, r);

  // Fundamental division identity. The historical bug returned q = 1, r = 0
  // for a >64-bit dividend over a single-limb divisor, which violates this.
  assert(q * y + r == x);

  // BigInt::div must agree with the independently implemented operators.
  assert(q == x / y);
  assert(r == x % y);

  return 0;
}
