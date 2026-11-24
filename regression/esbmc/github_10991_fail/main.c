#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  char d[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  unsigned off = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(off <= 8 && n <= 8 - off);
  memset(d + off, 7, n);
  unsigned j = nondet_uint();
  __ESBMC_assume(j < 8);
  assert(d[j] == 1);
  return 0;
}
