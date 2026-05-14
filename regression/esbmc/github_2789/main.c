// https://github.com/esbmc/esbmc/issues/2789
// C11 §6.5.7p3: shift by a negative amount is undefined behavior.
#include <assert.h>

int nondet_int();

int main()
{
  int a = nondet_int();
  assert((0 >> a) == 0);
  return 0;
}
