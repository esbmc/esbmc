#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  char c[16] = "abcdefghijklmno";
  unsigned p = nondet_uint(), q = nondet_uint(), k = nondet_uint();
  __ESBMC_assume(p < 8 && q < 8 && k >= 1 && k <= 8);
  memmove(c + p, c + q, k);
  assert(c[p + k - 1] == "abcdefghijklmno"[q + k - 1]);
  return 0;
}
