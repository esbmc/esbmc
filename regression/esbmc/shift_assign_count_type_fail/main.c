// Casting the count to the computation type is not merely redundant: it
// narrows a count of higher rank, so a count that is out of range for the
// promoted left operand can wrap into a valid one and the UB goes unreported.
// 2^32 truncates to 0 as int. #6924.
#include <assert.h>

int main(void)
{
  unsigned char u = 128;
  long long n = 4294967296LL;
  u >>= n;
  return u;
}
