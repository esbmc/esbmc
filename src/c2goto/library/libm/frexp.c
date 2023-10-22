
#include <math.h>
#include <stdint.h> /* uint64_t */
#include <string.h> /* memcpy */

double frexp(double x, int *exp)
{
  if(!isfinite(x))
    return x;
  if(x == 0.0)
  {
    *exp = 0;
    return x;
  }
  uint64_t v, m, z = 0;
  memcpy(&v, &x, sizeof(x));
  int e = (v >> 52) & ((1 << 11) - 1);
  m = v & (~z >> (64 - 52)); /* mantissa encoding */
  if(e == 0)
  { /* adjust for sub-normal numbers */
    for(int k = 0; k < 52; k++, e--, m <<= 1)
      if(m >> 52)
        break;
    m &= (1ULL << 52) - 1;
  }
  *exp = e - 1022;
  v &= 1ULL << 63;               /* keep only sign bit */
  v |= (uint64_t)1022 << 52 | m; /* add normalized exponent and mantissa */
  memcpy(&x, &v, sizeof(x));
  return x;
}

double ldexp(double x, int exp)
{
  if(!isfinite(x) || x == 0.0)
    return x;
  uint64_t v, m, z = 0;
  memcpy(&v, &x, sizeof(x));
  m = v & (~z >> (64 - 52)); /* mantissa encoding */
  int e = (v >> 52) & ((1 << 11) - 1);
  exp += e; /* add exponent encoding */
  if(exp < -52)
    return copysign(0.0, x);
  if(exp >= (1 << 11) - 1)
    return copysign(HUGE_VAL, x);
  if(exp < 0)
  {
    if(e)
      m |= 1ULL << 52;
    m >>= -exp;
    exp = 0;
  }
  v &= 1ULL << 63; /* keep only sign bit */
  m &= (1ULL << 52) - 1;
  v |= (uint64_t)exp << 52 | m; /* set exponent and mantissa */
  memcpy(&x, &v, sizeof(x));
  return x;
}
