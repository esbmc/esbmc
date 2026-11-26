// esbmc/esbmc#9227: one pointer copied byte by byte through a char buffer.
#include <string.h>
int x[6];
unsigned nondet_uint(void);
int main(void)
{
  int *p = &x[nondet_uint() % 6];
  unsigned char buf[sizeof p];
  int *q;
  memcpy(buf, &p, sizeof p);
  memcpy(&q, buf, sizeof p);
  __ESBMC_assert(q == p, "");
  return 0;
}
