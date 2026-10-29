#include <assert.h>

int main(void)
{
  _Bool b = 1;
  __ESBMC_assume(b != 0);
  b ^= 1;
  assert(b == 0);

  unsigned char c = 1;
  __ESBMC_assume(c == 1);
  c += 1;
  assert(c == 2);

  unsigned char w = 255;
  __ESBMC_assume(w == 255);
  w += 1;
  assert(w == 0);

  signed char s = nondet_char();
  __ESBMC_assume(s > 0 && s < 10);
  s -= 20;
  assert(s < 0);
  return 0;
}
