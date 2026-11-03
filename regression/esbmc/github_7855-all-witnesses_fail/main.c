#include <assert.h>
#include <stdint.h>
int nondet_int(void);
uintptr_t nondet_addr(void);
int main(void)
{
  int a[2] = {0, 1};
  uintptr_t u = nondet_addr();
  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x < 4);
  int *p = (int *)u;
  assert(p != (int *)((uintptr_t)a + sizeof(int)) || x != 1);
  return 0;
}
