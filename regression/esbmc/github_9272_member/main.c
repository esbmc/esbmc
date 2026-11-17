#include <stdint.h>

/* A sliced step takes s's address and the kept step only s.g[1]'s, after the
 * cast: s must still be registered before the cast for it to reach s.g[1]. */
uintptr_t nondet_uintptr(void);
struct S
{
  int f;
  int g[2];
};
int main(void)
{
  struct S s;
  uintptr_t us = (uintptr_t)&s;
  (void)us;
  uintptr_t x = nondet_uintptr();
  int *p = (int *)x;
  uintptr_t ug = (uintptr_t)&s.g[1];
  if (x == ug)
    __ESBMC_assert(p == &s.g[1], "cast of s.g[1]'s address is &s.g[1]");
  return 0;
}
