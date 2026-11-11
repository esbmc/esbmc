#include <assert.h>
int g[4];
int main(void)
{
  int *p = &g[0];
  unsigned char k = nondet_uchar();
  __ESBMC_assume(k < 2);
  ((unsigned char *)&p)[k] = 0x5a;
  assert(p == &g[0]);
  return 0;
}
