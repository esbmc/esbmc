#include <stdint.h>

/* A pointer read from a union member written with a local's address as an
 * integer is the local, even when the slicer drops every step taking its
 * address before the read (#10798). */
uintptr_t nondet_uintptr(void);
union U
{
  uintptr_t i;
  int *p;
};
int main(void)
{
  int a;
  uintptr_t ua = (uintptr_t)&a;
  union U u;
  u.i = nondet_uintptr();
  int *p = u.p;
  if (u.i == ua)
    __ESBMC_assert(p == &a, "union read of a's address is &a");
  return 0;
}
