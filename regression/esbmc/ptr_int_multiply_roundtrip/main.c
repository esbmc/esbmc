#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S { int x; int y; int z; } s = {.x = 1};
  uintptr_t u = (uintptr_t)&s;
  // Multiplying an address-derived integer used to lose object identity, so
  // the reconstructed pointer was rejected (#6545, fixed by #6905). Object
  // recovery is a may-approximation with a nondet offset -- see
  // docs/design/pointer-integer-provenance.md.
  u *= 2;
  u -= (uintptr_t)&s;
  int *p = (int *)u;
  *p = 3;
  assert(s.x == 3);
}
