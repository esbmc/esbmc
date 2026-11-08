#include <stdint.h>

/* #9273: under --smt-during-symex the cast is converted at the branch, before
 * &a first appears, and must still be able to land on a. */
int a;
uintptr_t nondet_uintptr(void);
_Bool nondet_bool(void);
int main(void)
{
  int *p = (int *)nondet_uintptr();
  uintptr_t ua = 1;
  if (nondet_bool())
    ua = (uintptr_t)&a;
  __ESBMC_assert(p != &a || ua == 0 || ua == 1, "p may point to a");
  return 0;
}
