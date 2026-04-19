
/* keep in sync with ../nextafterf-spec/main.c */

#include <math.h>
#include <errno.h>

#undef errno
extern int errno;

int main()
{
  double x = nondet_double();
  double y = nondet_double();
  double z = nextafter(x, y);

  /* any NaN argument <-> NaN result */
  assert((isnan(x) || isnan(y)) == (isnan(z) != 0));

  __ESBMC_assume(!isnan(z));
  double a = nondet_double();

  if(isless(x, y))
  { /* there is no double 'a' between x and z */
    assert(isnan(a) || islessequal(a, x) || islessequal(z, a));
    __ESBMC_assume(islessequal(z, a)); /* assume z <= a */
  }
  if(isgreater(x, y))
  { /* there is no double 'a' between x and z */
    assert(isnan(a) || islessequal(a, z) || islessequal(x, a));
    __ESBMC_assume(islessequal(a, z)); /* assume z >= a */
  }
  if(x == y)
    assert(x == z);
  if(isfinite(x))
  { /* |z| not finite -> |a| = inf and they have the same sign */
    assert(isfinite(z) || (isinf(a) && isinf(a) == isinf(z)));
  }
  if(isfinite(x) && !isfinite(z))
  {
    /* This is part of nextafter's behavior, but w/o a model of
     * feraiseexcept() and fetestexcept() we cannot prove it, yet.
    assert(fetestexcept(FE_OVERFLOW));
     */
    assert(errno == ERANGE);
  }
  if(islessgreater(x, y) && isfinite(z) && !isnormal(z))
  {
    /* This is part of nextafter's behavior, but w/o a model of
     * feraiseexcept() and fetestexcept() we cannot prove it, yet.
    assert(fetestexcept(FE_UNDERFLOW));
     */
    assert(errno == ERANGE);
  }
  if(isinf(x))
    assert(isinf(z) == isinf(x));

  /* This is part of nextafter's behavior, but w/o a model of
   * feraiseexcept() and fetestexcept() we cannot prove it, yet.
  assert(!fetestexcept(FE_UNDERFLOW) || !fetestexcept(FE_OVERFLOW));
   */
}
