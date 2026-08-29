#include <assert.h>
#include <stdint.h>

/* The bit pattern really is checked: asserting the wrong one must fail, so the
 * companion bitcast_intrinsic test cannot pass vacuously. */
int main()
{
  float f = 1.0f;
  uint32_t u;
  __ESBMC_bitcast(&u, &f);
  assert(u == 0x3f800001);
  return 0;
}
