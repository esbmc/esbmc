
#include <math.h>
#include <stdint.h> /* uint32_t */
#include <limits.h> /* CHAR_BIT */

/* compute x^n (with n != 0) */
static double pow_by_squaring(double x, uint32_t n)
{
  double result = 1.0;

  // clang-format off
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;

  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;

  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;

  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  if(n & 1) result *= x; x *= x; n >>= 1;
  // clang-format on

  return result;
}

double pow(double x, double y)
{
__ESBMC_HIDE:;
  if (x == 1.0 || y == 0.0)
    return 1.0;

  if (isnan(x))
    return x;
  if (isnan(y))
    return y;

  if (isinf(y))
  {
    if (x == -1.0)
      return 1.0;
    if ((fabs(x) < 1) != !signbit(y)) /* |x| < 1 && y < 0 || |x| > 1 && y > 0 */
      return INFINITY;
    return +0.0;
  }

  /* here we know y is finite */

  int ye;
  frexp(y, &ye);
  int is_int = nearbyint(y) == y;
  int odd_int = ye <= 53 && is_int && ((long long)y & 1);

  if (x == 0.0)
  {
    double r = signbit(y) ? HUGE_VAL : 0.0;
    return odd_int ? copysign(r, x) : r;
  }

  if (isinf(x))
  {
    double r = signbit(y) ? 0.0 : INFINITY;
    return odd_int ? copysign(r, x) : r;
  }

  /* here we know x is finite */

  if (!is_int && signbit(x))
    return NAN;

  if (is_int && ye <= 32) /* y is integral and fits into 32 bits */
  {
    int xe;
    if (fabs(frexp(x, &xe)) == 0.5) /* |x| = 2^(xe-1) is a power of 2 */
    {
      /* If y is too large to fit into the exponent, check whether |x| > 1 is
       * equivalent to y < 0. If so, 0.0 is the result, otherwise infinity.
       * In case y could fit into the exponent, ldexp() will do the necessary
       * under-/overflow checks. The multiplication (xe - 1) * (int)y cannot
       * overflow because (xe - 1) and (int)y both have at most 12 bits. */
      double r = ye > 12 ? (xe > 1) != !signbit(y) ? 0.0 : INFINITY
                         : ldexp(1.0, (xe - 1) * (int)y);
      return odd_int ? copysign(r, x) : r;
    }
    return pow_by_squaring(signbit(y) ? 1.0 / x : x, fabs(y));
  }

  double l = log(fabs(x));
  double r = exp(l * y);
  return odd_int ? copysign(r, x) : r;
}

double __pow(double base, double exponent)
{
__ESBMC_HIDE:;
  return pow(base, exponent);
}
