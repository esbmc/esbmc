#include <assert.h>

/* 1 + 2^-30 is not representable in float: the tie-free excess is rounded away
   under every mode except round-to-plus-infinity. */
int main()
{
  float x = 1.0f;
  float y = 0x1p-30f;
  float s = x + y;
  assert(s == 1.0f);
  return 0;
}
