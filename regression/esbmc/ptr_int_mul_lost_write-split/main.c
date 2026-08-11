// The missed-bug direction of esbmc/esbmc#6804, under --deref-unknown-objects:
// splitting the write over the tracked objects instead of absorbing it into the
// failed symbol recovers s, so the assertion is reported. Without the flag this
// is proved (see ptr_int_mul_lost_write, KNOWNBUG). gcc aborts on the assertion
// at -O0 and -O2. See docs/design/pointer-integer-provenance.md.

#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S
  {
    int x;
  } s = {.x = 42};

  uintptr_t u = (uintptr_t)&s;
  u *= 2;
  u -= (uintptr_t)&s; // u == (uintptr_t)&s

  int *p = (int *)u;
  *p = 3; // really writes s.x

  assert(s.x == 42);
  return 0;
}
