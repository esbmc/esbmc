#include <stdint.h>

/* #9273 with a function-local static, whose address only a later call takes. */
uintptr_t nondet_uintptr(void);
_Bool nondet_bool(void);
int *get(void)
{
  static int s;
  return &s;
}
int main(void)
{
  int *p = (int *)nondet_uintptr();
  int *q = 0;
  if (nondet_bool())
    q = get();
  __ESBMC_assert(p != q || q == 0, "p may point to s");
  return 0;
}
