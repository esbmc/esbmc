#include <math.h>

/* fmodf, not fmod: fmod lowers to the solver's exact fp.rem, and at double
 * width that term does not fit the suite's 8 GiB cap inside these quantifiers
 * under z3 (bitwuzla solves the double spelling in ~3s). Float width keeps a
 * libm call in the quantifier body, which is what this test is about. */
_Bool P(double x, double y, double z)
{
  double val = sin(y) + x + y;
  return fmodf((float)val, 3.14f) < 1.0f;
}

int main()
{
  double x, y, z;
  __ESBMC_assert(
    __ESBMC_forall(
      &x, __ESBMC_forall(&y, __ESBMC_exists(&z, P(x, 0.5, z)))),
    "nested quantifiers over libm calls");
}
