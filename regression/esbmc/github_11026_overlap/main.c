#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  const char *orig = "abcdefghijklmno";
  char c[16] = "abcdefghijklmno";
  unsigned p = nondet_uint(), q = nondet_uint(), k = nondet_uint();
  __ESBMC_assume(p < 8 && q < 8 && k <= 8);
  memmove(c + p, c + q, k);
  unsigned j = nondet_uint();
  __ESBMC_assume(j < 16);
  assert(c[j] == (p <= j && j < p + k ? orig[q + j - p] : orig[j]));
  return 0;
}
