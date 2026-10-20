#include <assert.h>
#include <stdio.h>

int main(void)
{
  char b[100];
  char x[3] = "ab";
  char y[5] = "abcd";
  char *p = nondet_int() ? x : y;

  int r = sprintf(b, "%s", p);
  assert(r == 2);
  return 0;
}
