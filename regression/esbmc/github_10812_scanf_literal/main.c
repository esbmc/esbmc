#include <assert.h>
#include <stdio.h>
int main()
{
  char buf[4] = "abc";
  char *p = nondet_bool() ? "abc" : buf;
  __ESBMC_assume(p == buf);
  sscanf("x", "%c", p);
  return 0;
}
