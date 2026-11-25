// #10887: flipping a payload bit of a NaN held in a float changes its bits.
#include <assert.h>
#include <stdint.h>
#include <string.h>

unsigned nondet_uint(void);

int main(void)
{
  uint32_t x = nondet_uint();
  __ESBMC_assume((x & 0x7f800000u) == 0x7f800000u && (x & 0x007fffffu) > 1);
  float f;
  memcpy(&f, &x, sizeof f);
  ((unsigned char *)&f)[0] ^= 1;
  uint32_t y;
  memcpy(&y, &f, sizeof y);
  assert(x == y);
  return 0;
}
