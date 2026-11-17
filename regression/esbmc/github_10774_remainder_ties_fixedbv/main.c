// A quotient exactly halfway between two integers rounds to the even one
// (C17 7.12.10.2), and remquo reports that quotient's sign and low bits.
#include <math.h>
int main()
{
  int q;
  __ESBMC_assert(remainder(5.0, 2.0) == 1.0, "2.5 rounds to 2");
  __ESBMC_assert(remainder(7.0, 2.0) == -1.0, "3.5 rounds to 4");
  __ESBMC_assert(remainder(-5.0, 2.0) == -1.0, "-2.5 rounds to -2");
  __ESBMC_assert(remainder(-7.0, 2.0) == 1.0, "-3.5 rounds to -4");
  __ESBMC_assert(remquo(7.0, 2.0, &q) == -1.0 && q == 4, "remquo 7, 2");
  __ESBMC_assert(remquo(-7.0, 2.0, &q) == 1.0 && q == -4, "remquo -7, 2");
  __ESBMC_assert(remquo(7.0, -2.0, &q) == -1.0 && q == -4, "remquo 7, -2");
  __ESBMC_assert(fmod(-7.5, 2.0) == -1.5, "fmod takes x's sign");
  __ESBMC_assert(fmodf(7.5f, -2.0f) == 1.5f, "fmodf");
  return 0;
}
