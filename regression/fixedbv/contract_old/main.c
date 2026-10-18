/* Function contracts over fixed-point types, including __ESBMC_old on a
 * fixed-typed global. The contracts pass normalizes additions inside
 * ensures clauses to IEEE_ADD for floats; fixed-point additions must stay
 * plain adds, or the ensures clause hands a fixed-sorted term to an FP
 * operation. */
#include <assert.h>

short _Fract acc = 0.25hr;

void bump(void)
{
  __ESBMC_ensures(acc == __ESBMC_old(acc) + 0.125hr);
  acc = acc + 0.125hr;
}

short _Fract halve(short _Fract x)
{
  __ESBMC_requires(x >= 0.0hr);
  __ESBMC_ensures(__ESBMC_return_value <= x);
  return x >> 1;
}

int main(void)
{
  bump();
  assert(acc == 0.375hr);
  assert(halve(0.5hr) == 0.25hr);
  return 0;
}
