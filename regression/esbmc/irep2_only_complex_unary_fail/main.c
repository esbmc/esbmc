#include <assert.h>
#include <complex.h>

double nondet_double(void);

int main(void)
{
  double complex z;
  __real__ z = nondet_double();
  __imag__ z = nondet_double();
  __ESBMC_assume(__imag__ z > 3.0 && __imag__ z < 4.0);

  /* Conjugation flips the sign of the imaginary part, so this is violated --
     a property broken through the lowering stays reportable on the component
     the lowering built. */
  double complex c = ~z;
  assert(__imag__ c == __imag__ z);

  return 0;
}
