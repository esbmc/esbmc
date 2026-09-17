/* The initialiser is truncated to the declared width, so it is not the
   int-width sum whenever that sum does not fit. */
#include <assert.h>

int nondet_int(void);

int main(void)
{
  _ExtInt(10) x = nondet_int();
  _ExtInt(10) y = nondet_int();
  _ExtInt(10) z = x + y;
  assert(z == (int)x + (int)y);
  return 0;
}
