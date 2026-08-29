// C11 6.5.7p3: a shift promotes its operands independently and takes its result
// from the left one, so the count of `E1 <<= E2` is not converted to the
// computation type the other compound assignments run in. The counts below keep
// their own rank; the results are unchanged by that. #6924.
#include <assert.h>

int main(void)
{
  short s = 4;
  s >>= 1;
  assert(s == 2);

  unsigned char c = 8;
  c >>= 2;
  assert(c == 2);

  long long l = 1024;
  l >>= 3;
  assert(l == 128);

  _Bool b = 1;
  b <<= 0;
  assert(b == 1);

  // A count whose rank is above the computation type still counts bits.
  unsigned char u = 128;
  long long n = 3;
  u >>= n;
  assert(u == 16);

  signed char sc = 64;
  short m = 2;
  sc >>= m;
  assert(sc == 16);

  return 0;
}
