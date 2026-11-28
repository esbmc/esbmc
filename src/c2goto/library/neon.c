/* Operational models of clang's AArch64 NEON load/store builtins.
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
