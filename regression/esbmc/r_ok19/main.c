#include <assert.h>

int main()
{
  char a[10];
  char b[5];
  char *p;

  int nondet;
  __ESBMC_assume(nondet == 1);
  if (nondet == 1) {
    p = a;
  }
  else {
    p = b;
  }

  assert(__ESBMC_r_ok(p, 6));
  return 0;
}
