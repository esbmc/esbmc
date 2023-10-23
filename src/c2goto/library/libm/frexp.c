
#include <math.h>
#include <stdint.h> /* uint64_t */
#include <string.h> /* memcpy */
#include <float.h>  /* *_MANT_DIG */

_Static_assert(FLT_RADIX == 2, "binary IEEE-754 float format");

#ifdef __FLOAT_WORD_ORDER__ /* Clang doesn't define __FLOAT_WORD_ORDER__ :( */
_Static_assert(
  __FLOAT_WORD_ORDER__ == __BYTE_ORDER__,
  "implementation via memcpy() assumes float and integer endianness matches");
#endif

#define DBL_EXP_BIAS 1023
#define DBL_MANT_BITS (DBL_MANT_DIG - 1)
#define DBL_EXP_BITS (64 - 1 - DBL_MANT_BITS)
#define DBL_EXP_MASK ((((uint64_t)1 << DBL_EXP_BITS) - 1) << DBL_MANT_BITS)

double frexp(double x, int *exp)
{
  if(!isfinite(x))
    return x;
  if(x == 0.0)
  {
    *exp = 0;
    return x;
  }
  int off = 0;
  if(!isnormal(x))
  {
    x *= 1ULL << DBL_MANT_BITS;
    off -= DBL_MANT_BITS;
  }
  uint64_t v;
  __ESBMC_bitcast(&v, &x);
  int e = (v & DBL_EXP_MASK) >> DBL_MANT_BITS;
  *exp = e - (DBL_EXP_BIAS - 1) + off;                /* range [0.5, 1) */
  v &= ~DBL_EXP_MASK;                                 /* clear exponent */
  v |= (uint64_t)(DBL_EXP_BIAS - 1) << DBL_MANT_BITS; /* set exponent to -1 */
  __ESBMC_bitcast(&x, &v);
  return x;
}

double ldexp(double x, int exp)
{
  if(!isfinite(x) || x == 0.0)
    return x;
  uint64_t v, m;
  __ESBMC_bitcast(&v, &x);
  m = v & (((uint64_t)1 << DBL_MANT_BITS) - 1); /* mantissa encoding */
  int e = (v & DBL_EXP_MASK) >> DBL_MANT_BITS;
  exp += e; /* add exponent encoding */
  if(exp < -DBL_MANT_BITS)
    return copysign(0.0, x);
  if(exp >= (1 << DBL_EXP_BITS) - 1)
    return copysign(HUGE_VAL, x);
  if(exp <= 0)
  {
    /* make a denormalized number */
    m |= 1ULL << DBL_MANT_BITS;
    m >>= 1 - exp;
    exp = 0;
  }
  v &= 1ULL << 63;                         /* keep only sign bit */
  v |= (uint64_t)exp << DBL_MANT_BITS | m; /* set exponent and mantissa */
  __ESBMC_bitcast(&x, &v);
  return x;
}
