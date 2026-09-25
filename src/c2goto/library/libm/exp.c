
#include <math.h>

/* How many terms the truncated Taylor series below expand to, between 2 and
 * ESBMC_FP_TAYLOR_MAX_TERMS. --fp-taylor-terms rewrites this initialiser
 * (esbmc/esbmc#2865): more terms buy accuracy at the cost of symex work, and
 * the useful setting is program-specific.
 *
 * The series are unrolled with a guard per term rather than looped: a loop
 * would consume the caller's --unwind budget and trip an unwinding assertion
 * inside the model. The guards fold away once the value is const-propagated. */
int __ESBMC_fp_taylor_terms = 8;

static double expm1_taylor(double x)
{
  /* Compute truncated Taylor series for e^x - 1 around 0:
   * x + x^2/2! + x^3/3! + x^4/4! + ... + x^n/n! */
  const int n = __ESBMC_fp_taylor_terms;
  double acc = x;
  double smd = x;
  acc += (smd *= x / 2);
  if (n >= 3)
    acc += (smd *= x / 3);
  if (n >= 4)
    acc += (smd *= x / 4);
  if (n >= 5)
    acc += (smd *= x / 5);
  if (n >= 6)
    acc += (smd *= x / 6);
  if (n >= 7)
    acc += (smd *= x / 7);
  if (n >= 8)
    acc += (smd *= x / 8);
  if (n >= 9)
    acc += (smd *= x / 9);
  if (n >= 10)
    acc += (smd *= x / 10);
  if (n >= 11)
    acc += (smd *= x / 11);
  if (n >= 12)
    acc += (smd *= x / 12);
  return acc;
}

double expm1(double x) /* exp(x) - 1 */
{
  switch (fpclassify(x))
  {
  case FP_NAN:
  case FP_ZERO:
    return x;
  case FP_INFINITE:
    return signbit(x) ? -1.0 : x;
  case FP_SUBNORMAL:
  case FP_NORMAL:
    break;
  }

  /* Taylor series converges everywhere, but the rate of convergence
   * is pretty bad; below we do a simple range reduction for larger |x|.
   */
  if (fabs(x) < 0x1p-3)
    return expm1_taylor(x);

  /* range reduction: exp(xm * 2^xe) = exp(xm) ^ (2^xe) */
  int xe;
  double xm = frexp(x, &xe); // |xm| in [2^-1,2^-0)
  xm *= 0x1p-3;
  xe += 3;                         // |xm| in [2^-4,2^-3)
  double r = expm1_taylor(xm) + 1; // r = exp(xm)
  /* xe is > 0 and xe < 1025+3, square xe times to account for 2^xe */
  int i = 0;
  while (i < xe)
  {
    r *= r;
    ++i;
  }

  return r - 1;
}

double exp(double x)
{
  return expm1(x) + 1;
}

static double log1p_taylor(double x)
{
  /* Compute truncated Taylor series of ln(x+1) around 0:
   * x - x^2/2 + x^3/3 - x^4/4 + ... +- x^n/n */
  const int n = __ESBMC_fp_taylor_terms;
  double acc = x;
  double smd = x;
  acc += (smd *= x) / -2;
  if (n >= 3)
    acc += (smd *= x) / 3;
  if (n >= 4)
    acc += (smd *= x) / -4;
  if (n >= 5)
    acc += (smd *= x) / 5;
  if (n >= 6)
    acc += (smd *= x) / -6;
  if (n >= 7)
    acc += (smd *= x) / 7;
  if (n >= 8)
    acc += (smd *= x) / -8;
  if (n >= 9)
    acc += (smd *= x) / 9;
  if (n >= 10)
    acc += (smd *= x) / -10;
  if (n >= 11)
    acc += (smd *= x) / 11;
  if (n >= 12)
    acc += (smd *= x) / -12;
  return acc;
}

double log1p(double x) /* ln(x+1) */
{
  switch (fpclassify(x))
  {
  case FP_NAN:
    return x;
  case FP_INFINITE:
    return signbit(x) ? NAN : x;
  case FP_ZERO:
    break;
  case FP_SUBNORMAL:
  case FP_NORMAL:
    if (x == -1.0)
      return -HUGE_VAL;
    if (x < -1.0)
      return NAN;
    break;
  }
  if (fabs(x) >= 0.125) /* adding 1 won't destroy many bits */
    return log(x + 1);

  return log1p_taylor(x);
}

double log(double x)
{
__ESBMC_HIDE:;
  return log2(x) * M_LN2;
}

double log2(double x)
{
  switch (fpclassify(x))
  {
  case FP_NAN:
    return x;
  case FP_INFINITE:
    return signbit(x) ? NAN : x;
  case FP_ZERO:
    return -HUGE_VAL;
  case FP_SUBNORMAL:
  case FP_NORMAL:
    if (signbit(x))
      return NAN;
    break;
  }

  int xe;
  double xm = frexp(x, &xe); /* xm in [0.5, 1) */
  if (xm < 2.0 / 3.0)
  {
    xm *= 2;
    xe--;
  }
  int n = 1; /* xm in [0.666..., 1.333...) */
  // clang-format off
  xm = sqrt(xm); n *= 2; /* xm in [0.816..., 1.154...) */
  xm = sqrt(xm); n *= 2; /* xm in [0.903..., 1.074...) */
  // xm = sqrt(xm); n *= 2;  /* xm in [0.950..., 1.036...) */
  // xm = sqrt(xm); n *= 2;  /* xm in [0.974..., 1.018...) */
  // clang-format on
  int xe2 = xe;
  return xe2 + n * log1p_taylor(xm - 1) / M_LN2;
}
