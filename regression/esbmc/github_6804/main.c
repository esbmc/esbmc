/* u is (uintptr_t)&s again, but the multiplication makes the value set
 * non-exhaustive, so no target resolves. The write then landed on the fallback
 * symbol, s.x kept 42, and this assertion -- false in C -- was proved. */
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
  u -= (uintptr_t)&s;

  int *p = (int *)u;
  *p = 3;

  assert(s.x == 42);
  return 0;
}
