#include <assert.h>

int main()
{
  char a[10];
  char b[5];
  char *p;

  int nondet;
  if (nondet == 1) {
    p = a;
  }
  else {
    p = b;
  }

  assert(__ESBMC_r_ok(p, nondet == 1 ? 6 : 3));
  return 0;
}
