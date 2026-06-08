#include <stdlib.h>
#include <assert.h>

int main()
{
  int addr1 = nondet_int();
  __ESBMC_assume(addr1 == 4);
  int addr2 = nondet_int();

  void *a = (void *)malloc(addr1);
  void *b = (void *)malloc(addr2);
  assert(0);
}
