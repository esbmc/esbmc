#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  unsigned x = 0x04030201;
  unsigned char b[4] = {0};
  unsigned i = nondet_uint(), n = nondet_uint();
  __ESBMC_assume(i < 4 && n >= 1 && n <= 4 - i);
  memcpy(b, (char *)&x + i, n);
  assert(b[0] == 1);

  return 0;
}
