/* Operational models of clang's AArch64 NEON builtins: loads and stores, lane
 * gets and sets, min/max and bitwise select.
 *
 * arm_neon.h lowers vld1q_u8, vst2_f32 and the rest to type-generic
 * __builtin_neon_* calls whose last argument is clang's NeonTypeFlags code:
 * bits 0-3 the element type, 0x10 unsigned, 0x20 a 128-bit (q) vector. The
 * frontend renames those calls to the __ESBMC_neon_* models below.
 *
 * Memory is accessed one lane at a time with the lane's own type, so memory
 * safety is checked per element as for memcpy, and a float lane is copied as a
 * float rather than byte by byte. LD1-4 and ST1-4 need no alignment, so a lane
 * at a misaligned offset in a byte buffer or another unpacked integer object
 * is moved as a packed one-member struct, whose access skips the alignment
 * check but keeps the bounds checks. A packed object still reports "Incorrect
 * alignment", and a misaligned float lane in a floating-point array reports
 * "Object accessed with incompatible base type" (see #11135). Registers are
 * accessed as whole vectors of that type. The moves are unrolled, not looped,
 * so they need no --unwind. */

#include <stdint.h>

typedef signed char __esbmc_neon_d __attribute__((vector_size(8)));
typedef signed char __esbmc_neon_q __attribute__((vector_size(16)));

#define __ESBMC_NEON_REP4(X, j) X(j) X((j) + 1) X((j) + 2) X((j) + 3)
#define __ESBMC_NEON_REP16(X, j)                                               \
  __ESBMC_NEON_REP4(X, j)                                                      \
  __ESBMC_NEON_REP4(X, (j) + 4)                                                \
  __ESBMC_NEON_REP4(X, (j) + 8) __ESBMC_NEON_REP4(X, (j) + 12)
#define __ESBMC_NEON_REP64(X)                                                  \
  __ESBMC_NEON_REP16(X, 0)                                                     \
  __ESBMC_NEON_REP16(X, 16)                                                    \
  __ESBMC_NEON_REP16(X, 32) __ESBMC_NEON_REP16(X, 48)

/* Register lanes come from, or go to, nvec vectors at reg: signed char
 * vectors when raw (a builtin's own vector arguments and results), otherwise
 * vectors of the lane type (the struct a multi-vector load writes). */
#define __ESBMC_NEON_REG_IN(k)                                                 \
  if (k < nvec)                                                                \
    t[k] = raw ? __builtin_bit_cast(V, ((Q *)reg)[k]) : ((V *)reg)[k];
#define __ESBMC_NEON_REG_OUT(k)                                                \
  if (k < nvec)                                                                \
  {                                                                            \
    if (raw)                                                                   \
      ((Q *)reg)[k] = __builtin_bit_cast(Q, t[k]);                             \
    else                                                                       \
      ((V *)reg)[k] = t[k];                                                    \
  }

/* Lane j % lanes of vector j / lanes sits in memory at j when contiguous and
 * at lane * nvec + vector when interleaved. */
#define __ESBMC_NEON_LANE(j)                                                   \
  if ((j) < nvec * lanes)                                                      \
  {                                                                            \
    int v = (j) / lanes, l = (j) % lanes;                                      \
    int i = interleaved ? l * nvec + v : (j);                                  \
    if (aligned)                                                               \
    {                                                                          \
      if (load)                                                                \
        t[v][l] = m[i];                                                        \
      else                                                                     \
        m[i] = t[v][l];                                                        \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      U u;                                                                     \
      if (load)                                                                \
      {                                                                        \
        u = um[i];                                                             \
        t[v][l] = u.x;                                                         \
      }                                                                        \
      else                                                                     \
      {                                                                        \
        u.x = t[v][l];                                                         \
        um[i] = u;                                                             \
      }                                                                        \
    }                                                                          \
  }

#define __ESBMC_NEON_MOVE_W(T, W)                                              \
  {                                                                            \
    typedef T V __attribute__((vector_size(W)));                               \
    typedef signed char Q __attribute__((vector_size(W)));                     \
    int lanes = W / sizeof(T);                                                 \
    typedef struct __attribute__((packed))                                     \
    {                                                                          \
      T x;                                                                     \
    } U;                                                                       \
    T *m = mem;                                                                \
    U *um = mem;                                                               \
    int aligned = __ESBMC_POINTER_OFFSET(mem) % sizeof(T) == 0;                \
    V t[4];                                                                    \
    if (!load)                                                                 \
    {                                                                          \
      __ESBMC_NEON_REP4(__ESBMC_NEON_REG_IN, 0)                                \
    }                                                                          \
    __ESBMC_NEON_REP64(__ESBMC_NEON_LANE)                                      \
    if (load)                                                                  \
    {                                                                          \
      __ESBMC_NEON_REP4(__ESBMC_NEON_REG_OUT, 0)                               \
    }                                                                          \
  }

#define __ESBMC_NEON_MOVE(T)                                                   \
  {                                                                            \
    if (type & 0x20)                                                           \
      __ESBMC_NEON_MOVE_W(T, 16)                                               \
    else                                                                       \
      __ESBMC_NEON_MOVE_W(T, 8)                                                \
  }

/* Move nvec vectors between memory at mem and registers at reg. */
void __ESBMC_neon_move(
  void *mem,
  void *reg,
  int type,
  int nvec,
  int interleaved,
  int load,
  int raw)
{
__ESBMC_HIDE:;
  int is_unsigned = type & 0x10;
  switch (type & 0xf)
  {
  case 0: /* Int8 */
    if (is_unsigned)
      __ESBMC_NEON_MOVE(uint8_t)
    else
      __ESBMC_NEON_MOVE(int8_t)
    break;
  case 1: /* Int16 */
    if (is_unsigned)
      __ESBMC_NEON_MOVE(uint16_t)
    else
      __ESBMC_NEON_MOVE(int16_t)
    break;
  case 2: /* Int32 */
    if (is_unsigned)
      __ESBMC_NEON_MOVE(uint32_t)
    else
      __ESBMC_NEON_MOVE(int32_t)
    break;
  case 3: /* Int64 */
    if (is_unsigned)
      __ESBMC_NEON_MOVE(uint64_t)
    else
      __ESBMC_NEON_MOVE(int64_t)
    break;
  case 4:  /* Poly8 */
  case 12: /* MFloat8, which ESBMC reads as an 8-bit unsigned integer */
    __ESBMC_NEON_MOVE(uint8_t)
    break;
  case 5: /* Poly16 */
    __ESBMC_NEON_MOVE(uint16_t)
    break;
  case 6: /* Poly64 */
    __ESBMC_NEON_MOVE(uint64_t)
    break;
  case 8: /* Float16 */
    __ESBMC_NEON_MOVE(_Float16)
    break;
  case 9: /* Float32 */
    __ESBMC_NEON_MOVE(float)
    break;
  case 10: /* Float64 */
    __ESBMC_NEON_MOVE(double)
    break;
  default:
    __ESBMC_assert(0, "unsupported NEON element type");
  }
}

__esbmc_neon_d __ESBMC_neon_vld1_v(const void *p, int type)
{
__ESBMC_HIDE:;
  __esbmc_neon_d r;
  __ESBMC_neon_move((void *)p, &r, type, 1, 0, 1, 1);
  return r;
}

__esbmc_neon_q __ESBMC_neon_vld1q_v(const void *p, int type)
{
__ESBMC_HIDE:;
  __esbmc_neon_q r;
  __ESBMC_neon_move((void *)p, &r, type, 1, 0, 1, 1);
  return r;
}

/* The multi-vector loads write their result through the first argument. */
#define __ESBMC_NEON_LOAD(name, nvec, interleaved)                             \
  void __ESBMC_neon_##name(void *ret, const void *p, int type)                 \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    __ESBMC_neon_move((void *)p, ret, type, nvec, interleaved, 1, 0);          \
  }

__ESBMC_NEON_LOAD(vld1_x2_v, 2, 0)
__ESBMC_NEON_LOAD(vld1q_x2_v, 2, 0)
__ESBMC_NEON_LOAD(vld1_x3_v, 3, 0)
__ESBMC_NEON_LOAD(vld1q_x3_v, 3, 0)
__ESBMC_NEON_LOAD(vld1_x4_v, 4, 0)
__ESBMC_NEON_LOAD(vld1q_x4_v, 4, 0)
__ESBMC_NEON_LOAD(vld2_v, 2, 1)
__ESBMC_NEON_LOAD(vld2q_v, 2, 1)
__ESBMC_NEON_LOAD(vld3_v, 3, 1)
__ESBMC_NEON_LOAD(vld3q_v, 3, 1)
__ESBMC_NEON_LOAD(vld4_v, 4, 1)
__ESBMC_NEON_LOAD(vld4q_v, 4, 1)

#define __ESBMC_NEON_STORE1(name, V)                                           \
  void __ESBMC_neon_##name(void *p, V a, int type)                             \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    __ESBMC_neon_move(p, &a, type, 1, 0, 0, 1);                                \
  }
#define __ESBMC_NEON_STORE2(name, V, interleaved)                              \
  void __ESBMC_neon_##name(void *p, V a, V b, int type)                        \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    V v[2] = {a, b};                                                           \
    __ESBMC_neon_move(p, v, type, 2, interleaved, 0, 1);                       \
  }
#define __ESBMC_NEON_STORE3(name, V, interleaved)                              \
  void __ESBMC_neon_##name(void *p, V a, V b, V c, int type)                   \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    V v[3] = {a, b, c};                                                        \
    __ESBMC_neon_move(p, v, type, 3, interleaved, 0, 1);                       \
  }
#define __ESBMC_NEON_STORE4(name, V, interleaved)                              \
  void __ESBMC_neon_##name(void *p, V a, V b, V c, V d, int type)              \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    V v[4] = {a, b, c, d};                                                     \
    __ESBMC_neon_move(p, v, type, 4, interleaved, 0, 1);                       \
  }

__ESBMC_NEON_STORE1(vst1_v, __esbmc_neon_d)
__ESBMC_NEON_STORE1(vst1q_v, __esbmc_neon_q)
__ESBMC_NEON_STORE2(vst1_x2_v, __esbmc_neon_d, 0)
__ESBMC_NEON_STORE2(vst1q_x2_v, __esbmc_neon_q, 0)
__ESBMC_NEON_STORE3(vst1_x3_v, __esbmc_neon_d, 0)
__ESBMC_NEON_STORE3(vst1q_x3_v, __esbmc_neon_q, 0)
__ESBMC_NEON_STORE4(vst1_x4_v, __esbmc_neon_d, 0)
__ESBMC_NEON_STORE4(vst1q_x4_v, __esbmc_neon_q, 0)
__ESBMC_NEON_STORE2(vst2_v, __esbmc_neon_d, 1)
__ESBMC_NEON_STORE2(vst2q_v, __esbmc_neon_q, 1)
__ESBMC_NEON_STORE3(vst3_v, __esbmc_neon_d, 1)
__ESBMC_NEON_STORE3(vst3q_v, __esbmc_neon_q, 1)
__ESBMC_NEON_STORE4(vst4_v, __esbmc_neon_d, 1)
__ESBMC_NEON_STORE4(vst4q_v, __esbmc_neon_q, 1)

/* vget_lane_u8, vsetq_lane_f32 and the rest lower to one builtin per lane
 * size, with the vector and the scalar typed as in clang's prototype. clang
 * has already checked that the lane is a constant in range. */
#define __ESBMC_NEON_LANE_GET_SET(q, t, R, T, W)                               \
  typedef T __esbmc_neon_##q##t __attribute__((vector_size(W)));               \
  R __ESBMC_neon_vget##q##_lane_##t(__esbmc_neon_##q##t v, int lane)           \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    return v[lane];                                                            \
  }                                                                            \
  __esbmc_neon_##q##t __ESBMC_neon_vset##q##_lane_##t(                         \
    T x, __esbmc_neon_##q##t v, int lane)                                      \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    v[lane] = x;                                                               \
    return v;                                                                  \
  }

__ESBMC_NEON_LANE_GET_SET(, i8, unsigned char, signed char, 8)
__ESBMC_NEON_LANE_GET_SET(q, i8, unsigned char, signed char, 16)
__ESBMC_NEON_LANE_GET_SET(, i16, unsigned short, short, 8)
__ESBMC_NEON_LANE_GET_SET(q, i16, unsigned short, short, 16)
__ESBMC_NEON_LANE_GET_SET(, i32, unsigned int, int, 8)
__ESBMC_NEON_LANE_GET_SET(q, i32, unsigned int, int, 16)
__ESBMC_NEON_LANE_GET_SET(, i64, uint64_t, int64_t, 8)
__ESBMC_NEON_LANE_GET_SET(q, i64, uint64_t, int64_t, 16)
__ESBMC_NEON_LANE_GET_SET(, f32, float, float, 8)
__ESBMC_NEON_LANE_GET_SET(q, f32, float, float, 16)
__ESBMC_NEON_LANE_GET_SET(, f64, double, double, 8)
__ESBMC_NEON_LANE_GET_SET(q, f64, double, double, 16)

/* Lane j of the result is the larger (max) or smaller lane j of x and y. A
 * float lane follows FMAX and FMIN: a NaN in either lane gives a NaN, and of
 * two equal lanes, which differ at most in the sign of zero, max takes the
 * one with a clear sign bit and min the one with a set sign bit. */
#define __ESBMC_NEON_MINMAX_LANE(j)                                            \
  if ((j) < lanes)                                                             \
  {                                                                            \
    if (x[j] != x[j] || y[j] != y[j])                                          \
      z[j] = x[j] + y[j];                                                      \
    else if (x[j] == y[j])                                                     \
      z[j] = (max == !!__builtin_signbit(x[j])) ? y[j] : x[j];                 \
    else                                                                       \
      z[j] = (max ? x[j] > y[j] : x[j] < y[j]) ? x[j] : y[j];                  \
  }
#define __ESBMC_NEON_INT_MINMAX_LANE(j)                                        \
  if ((j) < lanes)                                                             \
    z[j] = (max ? x[j] > y[j] : x[j] < y[j]) ? x[j] : y[j];

#define __ESBMC_NEON_MINMAX_W(T, W, LANE)                                      \
  {                                                                            \
    typedef T V __attribute__((vector_size(W)));                               \
    typedef signed char Q __attribute__((vector_size(W)));                     \
    int lanes = W / sizeof(T);                                                 \
    V x = __builtin_bit_cast(V, *(Q *)a), y = __builtin_bit_cast(V, *(Q *)b);  \
    V z;                                                                       \
    __ESBMC_NEON_REP16(LANE, 0)                                                \
    *(Q *)r = __builtin_bit_cast(Q, z);                                        \
  }

#define __ESBMC_NEON_MINMAX(T, LANE)                                           \
  {                                                                            \
    if (type & 0x20)                                                           \
      __ESBMC_NEON_MINMAX_W(T, 16, LANE)                                       \
    else                                                                       \
      __ESBMC_NEON_MINMAX_W(T, 8, LANE)                                        \
  }

/* Write to r the lane-wise maximum, or minimum, of the vectors at a and b. */
void __ESBMC_neon_minmax(
  void *r,
  const void *a,
  const void *b,
  int type,
  int max)
{
__ESBMC_HIDE:;
  int is_unsigned = type & 0x10;
  switch (type & 0xf)
  {
  case 0: /* Int8 */
    if (is_unsigned)
      __ESBMC_NEON_MINMAX(uint8_t, __ESBMC_NEON_INT_MINMAX_LANE)
    else
      __ESBMC_NEON_MINMAX(int8_t, __ESBMC_NEON_INT_MINMAX_LANE)
    break;
  case 1: /* Int16 */
    if (is_unsigned)
      __ESBMC_NEON_MINMAX(uint16_t, __ESBMC_NEON_INT_MINMAX_LANE)
    else
      __ESBMC_NEON_MINMAX(int16_t, __ESBMC_NEON_INT_MINMAX_LANE)
    break;
  case 2: /* Int32 */
    if (is_unsigned)
      __ESBMC_NEON_MINMAX(uint32_t, __ESBMC_NEON_INT_MINMAX_LANE)
    else
      __ESBMC_NEON_MINMAX(int32_t, __ESBMC_NEON_INT_MINMAX_LANE)
    break;
  case 9: /* Float32 */
    __ESBMC_NEON_MINMAX(float, __ESBMC_NEON_MINMAX_LANE)
    break;
  case 10: /* Float64 */
    __ESBMC_NEON_MINMAX(double, __ESBMC_NEON_MINMAX_LANE)
    break;
  default:
    __ESBMC_assert(0, "unsupported NEON element type");
  }
}

#define __ESBMC_NEON_MINMAX_FN(name, V, max)                                   \
  V __ESBMC_neon_##name(V a, V b, int type)                                    \
  {                                                                            \
  __ESBMC_HIDE:;                                                               \
    V r;                                                                       \
    __ESBMC_neon_minmax(&r, &a, &b, type, max);                                \
    return r;                                                                  \
  }

__ESBMC_NEON_MINMAX_FN(vmax_v, __esbmc_neon_d, 1)
__ESBMC_NEON_MINMAX_FN(vmaxq_v, __esbmc_neon_q, 1)
__ESBMC_NEON_MINMAX_FN(vmin_v, __esbmc_neon_d, 0)
__ESBMC_NEON_MINMAX_FN(vminq_v, __esbmc_neon_q, 0)

/* vbslq_u8 and the rest pick each bit from a where the mask m is set and from
 * b where it is clear, whatever the element type. */
__esbmc_neon_d __ESBMC_neon_vbsl_v(
  __esbmc_neon_d m,
  __esbmc_neon_d a,
  __esbmc_neon_d b,
  int type)
{
__ESBMC_HIDE:;
  return (m & a) | (~m & b);
}

__esbmc_neon_q __ESBMC_neon_vbslq_v(
  __esbmc_neon_q m,
  __esbmc_neon_q a,
  __esbmc_neon_q b,
  int type)
{
__ESBMC_HIDE:;
  return (m & a) | (~m & b);
}
