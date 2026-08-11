// A write through a pointer recovered from a multiplicative round trip is
// absorbed by the failed symbol that dereference() builds for a non-exhaustive
// value set, so asserting the write did not happen is proved. False in C: gcc
// aborts on this assertion at -O0 and -O2. See
// docs/design/pointer-integer-provenance.md (esbmc/esbmc#6545).

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
