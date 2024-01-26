
#include <math.h>

static double expm1_taylor(double x)
{
  /* Compute truncated Taylor series for e^x - 1 around 0:
   * x + x^2/2! + x^3/3! + x^4/4! + ... + x^n/n! */
  double acc = x;
  double smd = x;
  acc += (smd *= x / 2);
  acc += (smd *= x / 3);
  acc += (smd *= x / 4);
  acc += (smd *= x / 5);
  acc += (smd *= x / 6);
  // acc += (smd *= x / 7);
  // acc += (smd *= x / 8);
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
  for (int i = 0; i < xe; i++)
    r *= r;
  return r - 1;
}

double exp(double x)
{
  return expm1(x) + 1;
}

static double log1p_taylor(double x)
{
  /* Compute truncated Taylor series of ln(x+1) around 0:
   * x - x^2/2! + x^3/3! - x^4/4! + ... +- x^n/n! */
  double acc = x;
  double smd = x;
  acc += (smd *= x / -2);
  acc += (smd *= x / -3);
  acc += (smd *= x / -4);
  acc += (smd *= x / -5);
  // acc += (smd *= x / -6);
  // acc += (smd *= x / -7);
  // acc += (smd *= x / -8);
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
