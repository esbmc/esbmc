#include <stdio.h>

int nondet_int(void);

int main(void)
{
  char b[4];
  sprintf(b, "%d", nondet_int());
  return 0;
}
