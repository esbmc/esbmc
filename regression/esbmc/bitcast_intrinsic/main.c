#include <assert.h>
#include <stdint.h>

/* __ESBMC_bitcast(tgt, src) reinterprets *src as the type of *tgt, for equal
 * pointee widths. Pins the example documented in docs/constructs. */
int main()
{
  float f = 1.0f;
  uint32_t u;
  __ESBMC_bitcast(&u, &f);
  assert(u == 0x3f800000);

  double d = 2.0;
  uint64_t v;
  __ESBMC_bitcast(&v, &d);
  assert(v == 0x4000000000000000ULL);

  float back;
  __ESBMC_bitcast(&back, &u);
  assert(back == 1.0f);
  return 0;
}
