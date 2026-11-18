#include <stdint.h>

/* A pointer read from a union member written with a nondet integer can name
 * any object, b included, with or without slicing (#10798). */
uintptr_t nondet_uintptr(void);
union U
{
  uintptr_t i;
  int *p;
};
int main(void)
{
  int a = 1, b = 2;
  int *q = &a;
  (void)q;
  union U u;
  u.i = nondet_uintptr();
  int *p = u.p;
  __ESBMC_assert(
    __ESBMC_POINTER_OBJECT(p) <= 1 || __ESBMC_same_object(p, &b),
    "p names null, invalid or b");
  return 0;
}
