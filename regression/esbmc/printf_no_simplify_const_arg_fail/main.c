#include <assert.h>
#include <stdio.h>

int main()
{
  /* Non-vacuity control: the return value is pinned to the exact length, so
     asserting the wrong length must fail rather than pass on a widened range. */
  int a = printf("%02X", (unsigned)0xAB);
  assert(a == 3);
}
