#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

int main()
{
  unsigned long long x = 0x0807060504030201ULL;
  unsigned char buf[8] = {0};
  unsigned s = nondet_uint(), m = nondet_uint();
  __ESBMC_assume(s < 8 && m >= 1 && m <= 8 - s);
  memcpy(buf, (unsigned char *)&x + s, m);
  assert(buf[0] == s + 1);
  return 0;
}
