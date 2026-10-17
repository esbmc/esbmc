#include <assert.h>
int nondet_int(void);
int main(void)
{
  int value = nondet_int();
  __ESBMC_assume(value == 1);
  assert(value == 0);
  assert(value == 1);
}
