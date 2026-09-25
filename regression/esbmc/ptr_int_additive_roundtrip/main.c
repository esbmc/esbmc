#include <stdint.h>
#include <assert.h>

int main(void)
{
  struct S { int x; int y; int z; } s = {.x = 1};
  uintptr_t u = (uintptr_t)&s;
  // Additive arithmetic on an address-derived integer keeps object identity;
  // pins the boundary of #6545 rather than the integer round-trip alone.
  u += 4;
  u -= 4;
  int *p = (int *)u;
  *p = 3;
  assert(s.x == 3);
}
