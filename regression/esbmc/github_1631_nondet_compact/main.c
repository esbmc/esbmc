#include <stdlib.h>
#include <assert.h>

int main()
{
  int addr2 = nondet_int();
  __ESBMC_assume(addr2 < 0);
  void *b = (void *)malloc(addr2);
  assert(0);
}
