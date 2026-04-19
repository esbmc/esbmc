#include <assert.h>

int main()
{
  char a[10];
  char b[5];
  char *p;

  int nondet;
   if (nondet > 20) {
    p = a;
  }
  else {
    p = b;
  }

  assert(__ESBMC_r_ok(p, 6));
  return 0;
}
