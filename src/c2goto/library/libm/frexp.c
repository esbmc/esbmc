
#include <fenv.h>
#include <math.h>
#include <stdint.h> /* uint64_t */
#include <float.h>  /* *_MANT_DIG */
#include <limits.h> /* INT_(MIN|MAX) */

_Static_assert(FLT_RADIX == 2, "binary IEEE-754 float format");

#ifdef __FLOAT_WORD_ORDER__ /* Clang doesn't define __FLOAT_WORD_ORDER__ :( */
_Static_assert(
  __FLOAT_WORD_ORDER__ == __BYTE_ORDER__,
  "implementation via memcpy() assumes float and integer endianness matches");
#endif

#define FLT_BITS 32
#define FLT_EXP_BITS 8
#define FLT_MANT_BITS (FLT_MANT_DIG - 1)

#define DBL_BITS 64
#define DBL_EXP_BITS 11
#define DBL_MANT_BITS (DBL_MANT_DIG - 1)

/* No support for frexpl() and ldexpl() on:
 *
 * 32-bit x86 has __SIZEOF_LONG_DOUBLE__ == 12 and we have no compatible integer
 * type of the same size to work with.
 *
 * PowerPC defaults to "extended IBM long double", which are double-double
 * numbers and not IEEE-conformant; this default can be turned off via the
 * -mabi=ieeelongdouble switch.
 */
#if __SIZEOF_LONG_DOUBLE__ == __SIZEOF_INT128__ &&                             \
  !defined(__LONG_DOUBLE_IBM128__)
#  define LDBL_BITS 128
#  define LDBL_EXP_BITS 15
/* ESBMC encodes a 128-bit long double as IEEE binary128 even where
 * LDBL_MANT_DIG describes the x87 format (64 on x86-64). */
#  define LDBL_MANT_BITS 112
typedef __uint128_t __UINT128_TYPE__;
#elif __SIZEOF_LONG_DOUBLE__ == 8
/* long double is double (AArch64 Darwin, Windows). */
#  define LDBL_BITS 64
#  define LDBL_EXP_BITS 11
#  define LDBL_MANT_BITS 52
#endif

#define TYPE1(n) __UINT##n##_TYPE__
#define TYPE0(n) TYPE1(n)
#define BITS(pre) pre##_BITS
#define TYPE(pre) TYPE0(BITS(pre))
#define EXP_BIAS(pre) ((pre##_MAX_EXP - pre##_MIN_EXP + 1) / 2)
#define MANT_BITS(pre) (pre##_MANT_BITS)
#define EXP_BITS(pre) (pre##_EXP_BITS)
#define EXP_MASK(pre) ((((TYPE(pre))1 << EXP_BITS(pre)) - 1) << MANT_BITS(pre))

#define FREXP(name, type, pre)                                                 \
  type name(type x, int *exp)                                                  \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (!isfinite(x))                                                          \
      return x;                                                                \
    if (x == 0)                                                                \
    {                                                                          \
      *exp = 0;                                                                \
      return x;                                                                \
    }                                                                          \
    int off = 0;                                                               \
    if (!isnormal(x))                                                          \
    {                                                                          \
      x *= (TYPE(pre))1 << MANT_BITS(pre);                                     \
      off -= MANT_BITS(pre);                                                   \
    }                                                                          \
    TYPE(pre) v;                                                               \
    __ESBMC_bitcast(&v, &x);                                                   \
    int e = (v & EXP_MASK(pre)) >> MANT_BITS(pre);                             \
    *exp = e - (EXP_BIAS(pre) - 1) + off; /* range [0.5, 1) */                 \
    v &= ~EXP_MASK(pre);                  /* clear exponent */                 \
    v |= (TYPE(pre))(EXP_BIAS(pre) - 1)                                        \
         << MANT_BITS(pre); /* set exponent to -1 */                           \
    __ESBMC_bitcast(&x, &v);                                                   \
    return x;                                                                  \
  }

#define LDEXP(name, type, pre, suff, SUFF)                                     \
  type name(type x, int exp)                                                   \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    if (!isfinite(x) || x == 0.0##suff)                                        \
      return x;                                                                \
    if (!isnormal(x))                                                          \
    {                                                                          \
      x *= (TYPE(pre))1 << MANT_BITS(pre);                                     \
      exp = exp < INT_MIN + MANT_BITS(pre) ? INT_MIN : exp - MANT_BITS(pre);   \
    }                                                                          \
    TYPE(pre) v;                                                               \
    __ESBMC_bitcast(&v, &x);                                                   \
    const TYPE(pre) sign = v & (TYPE(pre))1 << (BITS(pre) - 1);                \
    TYPE(pre) m = v & (((TYPE(pre))1 << MANT_BITS(pre)) - 1);                  \
    int e = (v & EXP_MASK(pre)) >> MANT_BITS(pre);                             \
    const int rm = fegetround();                                               \
    if (exp >= (1 << EXP_BITS(pre)) - 1 - e)                                   \
    {                                                                          \
      /* overflow: infinity, or the largest finite value when rounding away */ \
      const int to_max = rm == FE_TOWARDZERO || (rm == FE_UPWARD && sign) ||   \
                         (rm == FE_DOWNWARD && !sign);                         \
      v = sign | (EXP_MASK(pre) - to_max);                                     \
    }                                                                          \
    else if (exp > -e)                                                         \
      v = sign | (TYPE(pre))(exp + e) << MANT_BITS(pre) | m;                   \
    else                                                                       \
    {                                                                          \
      /* subnormal: shift out 1 - (exp + e) bits and round them in the current \
       * mode. Below half the least subnormal every exponent rounds alike. */  \
      const int shift =                                                        \
        exp < -MANT_BITS(pre) - 1 - e ? MANT_BITS(pre) + 2 : 1 - (exp + e);    \
      m |= (TYPE(pre))1 << MANT_BITS(pre);                                     \
      const TYPE(pre) rest = m & (((TYPE(pre))1 << shift) - 1);                \
      const TYPE(pre) half = (TYPE(pre))1 << (shift - 1);                      \
      m >>= shift;                                                             \
      if (                                                                     \
        rm == FE_TONEAREST  ? rest > half || (rest == half && (m & 1))         \
        : rm == FE_UPWARD   ? rest && !sign                                    \
        : rm == FE_DOWNWARD ? rest && sign                                     \
                            : 0)                                               \
        ++m; /* a carry into the exponent gives the least normal value */      \
      v = sign | m;                                                            \
    }                                                                          \
    __ESBMC_bitcast(&x, &v);                                                   \
    return x;                                                                  \
  }

FREXP(frexpf, float, FLT)
LDEXP(ldexpf, float, FLT, f, F)

FREXP(frexp, double, DBL)
LDEXP(ldexp, double, DBL, , )

#ifdef LDBL_BITS
FREXP(frexpl, long double, LDBL)
LDEXP(ldexpl, long double, LDBL, l, L)
#endif

/* we asserted this above for the definitions to work, but if they change let's
 * better be safe */
#if FLT_RADIX == 2
#  define SCALBN(type, suff)                                                   \
    type scalbn##suff(type x, int exp)                                         \
    {                                                                          \
      return ldexp##suff(x, exp);                                              \
    }
SCALBN(float, f)
SCALBN(double, )
SCALBN(long double, l)

#  define SCALBLN(type, suff, SUFF)                                            \
    type scalbln##suff(type x, long exp)                                       \
    {                                                                          \
      exp = exp > INT_MAX ? INT_MAX : exp < INT_MIN ? INT_MIN : exp;           \
      return ldexp##suff(x, exp);                                              \
    }
SCALBLN(float, f, F)
SCALBLN(double, , )
SCALBLN(long double, l, L)

#else /* FLT_RADIX != 2 */
/* TODO: missing implementation */
#endif
