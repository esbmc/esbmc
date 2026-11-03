#include <stdio.h>

int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  __ESBMC_assume(x >= 10 && x <= 99);
  char d[2];
  sprintf(d, "%d", x);
  return 0;
}
