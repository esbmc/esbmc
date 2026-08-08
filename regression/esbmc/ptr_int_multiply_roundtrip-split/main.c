#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S { int x; int y; int z; } s = {.x = 1};
  uintptr_t u = (uintptr_t)&s;
  // The spurious-counterexample direction of esbmc/esbmc#6804. Multiplying an
  // address-derived integer loses object identity, so without
  // --deref-unknown-objects the reconstructed pointer is rejected and this
  // reports FAILED (see ptr_int_multiply_roundtrip, KNOWNBUG). Splitting the
  // write over the tracked objects recovers s and the assertion holds.
  u *= 2;
  u -= (uintptr_t)&s;
  int *p = (int *)u;
  *p = 3;
  assert(s.x == 3);
}
