#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S { int x; int y; int z; } s = {.x = 1};
  uintptr_t u = (uintptr_t)&s;
  // Multiplying an address-derived integer loses object identity, so the
  // reconstructed pointer is rejected and this reports FAILED. Known
  // limitation; docs/design/pointer-integer-provenance.md records why the
  // obvious fix is unsound (#6545). The additive round-trip in
  // ptr_int_additive_roundtrip is tracked.
  u *= 2;
  u -= (uintptr_t)&s;
  int *p = (int *)u;
  *p = 3;
  assert(s.x == 3);
}
