#include <assert.h>
#include <stdio.h>

/* #10811: only the object p points to may be written. */
_Bool nondet_bool(void);
int a[2] = {1, 2};
int b[2];

int main()
{
  int *p = nondet_bool() ? a : b;
  __ESBMC_assume(p == a);
  __ESBMC_init_object(p);
  assert(a[0] == 1);
  return 0;
}
