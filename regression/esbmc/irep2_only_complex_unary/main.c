#include <assert.h>
#include <complex.h>

int nondet_int(void);
double nondet_double(void);

int main(void)
{
  /* Operands are nondet and the expected values are scalar expressions over
     the components, so a wrong per-component formula is not restated on both
     sides of the comparison. The assumed ranges also exclude NaN, which no
     equality would hold over. */
  double complex z;
  __real__ z = nondet_double();
  __imag__ z = nondet_double();
  double zr = __real__ z, zi = __imag__ z;
  __ESBMC_assume(zr > 1.0 && zr < 2.0 && zi > 3.0 && zi < 4.0);

  double complex n = -z;
  assert(__real__ n == -zr && __imag__ n == -zi);
  double complex c = ~z;
  assert(__real__ c == zr && __imag__ c == -zi);

  /* A typedef'd complex is canonicalised before the arm sees it. */
  typedef double complex cdouble;
  cdouble t = -z;
  assert(__real__ t == -zr && __imag__ t == -zi);

  /* The operand here is the binary arm's output, so the components are read
     out of a struct literal rather than a symbol. Component order follows the
     lowering, so the two sides round identically. */
  double complex q = -(z * z);
  assert(__real__ q == -(zr * zr - zi * zi));
  assert(__imag__ q == -(zr * zi + zi * zr));

  /* The integral element type reaches the same arm. */
  int complex a;
  __real__ a = nondet_int();
  __imag__ a = nondet_int();
  int ar = __real__ a, ai = __imag__ a;
  __ESBMC_assume(ar >= 1 && ar <= 4 && ai >= 1 && ai <= 4);

  int complex m = -a;
  assert(__real__ m == -ar && __imag__ m == -ai);
  int complex k = ~a;
  assert(__real__ k == ar && __imag__ k == -ai);

  return 0;
}
