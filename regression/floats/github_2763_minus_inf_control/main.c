#include <assert.h>

/* -1 - 2^-30 rounds away from zero only under round-to-minus-infinity. */
int main()
{
  float x = -1.0f;
  float y = -0x1p-30f;
  float s = x + y;
  assert(s == -1.0f);
  return 0;
}
