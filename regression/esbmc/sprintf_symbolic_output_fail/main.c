#include <stdio.h>

unsigned nondet_uint(void);

int main(void)
{
  char b[4];
  sprintf(b, "%x", nondet_uint());
  return 0;
}
