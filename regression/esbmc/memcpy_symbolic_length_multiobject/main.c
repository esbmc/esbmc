/* The symbolic-length model needs one concrete object per operand. A pointer
   with two candidates declines to the C byte loop, which must still be
   correct. */
#include <string.h>
#include <assert.h>

int main()
{
  char a[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  char b[8] = {2, 2, 2, 2, 2, 2, 2, 2};
  char d[8] = {0};

  unsigned long i = nondet_ulong();
  __ESBMC_assume(i < 2);
  char *s = i ? a : b;

  unsigned long n = nondet_ulong();
  __ESBMC_assume(n == 4);

  memcpy(d, s, n);
  assert(d[0] == 1 || d[0] == 2);
  assert(d[4] == 0);
  return 0;
}
