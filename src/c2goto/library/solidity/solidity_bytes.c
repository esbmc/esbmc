/* Solidity bytes type operations */
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include "solidity_types.h"

typedef struct BytesPool
{
  unsigned char *pool;
  size_t pool_cursor;
} BytesPool;

typedef struct BytesStatic
{
  unsigned char data[32];
  size_t length;
} BytesStatic;

/* Hard cap for BytesStatic loop unrolling. Solidity's bytes1..bytes32
 * never exceed 32 bytes, so using the static array size as the concrete
 * loop bound lets symex unwind deterministically even when .length is
 * symbolic (e.g. when a BytesStatic flows in from a nondet parameter). */
#define _ESBMC_BYTES_STATIC_MAX 32

/* Compile-time 32-step unroll. Each STEP(i) macro receives a literal
 * constant i so b->data[i] resolves to a concrete memory offset and
 * symex never has to unwind a symbolic-bound loop.  Used by every
 * BytesStatic helper that previously had a `for (i < 32; if (i >=
 * length) break)` loop — that pattern relied on `--unwind` to
 * statically resolve, which (a) flooded stderr with "Not unwinding"
 * messages on every visit and (b) silently truncated the result for
 * `--unwind < length`. */
#define _ESBMC_BS_UNROLL_32(STEP)                                              \
  STEP(0);                                                                     \
  STEP(1);                                                                     \
  STEP(2);                                                                     \
  STEP(3);                                                                     \
  STEP(4);                                                                     \
  STEP(5);                                                                     \
  STEP(6);                                                                     \
  STEP(7);                                                                     \
  STEP(8);                                                                     \
  STEP(9);                                                                     \
  STEP(10);                                                                    \
  STEP(11);                                                                    \
  STEP(12);                                                                    \
  STEP(13);                                                                    \
  STEP(14);                                                                    \
  STEP(15);                                                                    \
  STEP(16);                                                                    \
  STEP(17);                                                                    \
  STEP(18);                                                                    \
  STEP(19);                                                                    \
  STEP(20);                                                                    \
  STEP(21);                                                                    \
  STEP(22);                                                                    \
  STEP(23);                                                                    \
  STEP(24);                                                                    \
  STEP(25);                                                                    \
  STEP(26);                                                                    \
  STEP(27);                                                                    \
  STEP(28);                                                                    \
  STEP(29);                                                                    \
  STEP(30);                                                                    \
  STEP(31)

void bytes_dynamic_init_check(const int initialized)
{
__ESBMC_HIDE:;
  if (initialized == 0)
    assert(!"Uninitialized Dynamic Bytes");
}

void bytes_dynamic_bounds_check(size_t index, size_t length)
{
__ESBMC_HIDE:;
  if (index >= length)
    assert(!"Out-of-bounds access on Dynamic Bytes");
}

unsigned char hex_char_to_nibble(char c)
{
__ESBMC_HIDE:;
  if ('0' <= c && c <= '9')
    return c - '0';
  else if ('a' <= tolower(c) && tolower(c) <= 'f')
    return tolower(c) - 'a' + 10;
  else
    abort();
  return 0;
}

BytesStatic bytes_static_from_hex(const char *hex_str, size_t len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
  size_t hex_len = len - 2;
  b.length = hex_len / 2;
#define STEP(i)                                                                \
  do                                                                           \
  {                                                                            \
    if (b.length > (i))                                                        \
    {                                                                          \
      unsigned char _high = hex_char_to_nibble(hex_str[2 + (i) * 2]);          \
      unsigned char _low = hex_char_to_nibble(hex_str[2 + (i) * 2 + 1]);       \
      b.data[i] = (_high << 4) | _low;                                         \
    }                                                                          \
  } while (0)
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  return b;
}

BytesStatic bytes_static_from_string(const char *str, size_t len)
{
__ESBMC_HIDE:;
  /* Constant-index unroll instead of memcpy/memset — generic memcpy/
   * memset decay to symbolic-bound `__memcpy_impl`/`__memset_impl`
   * byte-loops that k-induction unwinds 1..32 times per call,
   * blowing up the VCC count under bytes16/bytes32 init. Same
   * pattern as `bytes_static_from_hex` (line 63). Pinned by
   * `bytes_1` (90s solver timeout pre-fix). */
  BytesStatic b = {0};
  if (len > 32)
    len = 32;
  size_t copy_len = strlen(str);
  if (copy_len > len)
    copy_len = len;
  b.length = len;
#define STEP(i)                                                                \
  do                                                                           \
  {                                                                            \
    if ((i) < copy_len)                                                        \
      b.data[i] = (unsigned char)str[i];                                       \
    else if ((i) < len)                                                        \
      b.data[i] = 0;                                                           \
  } while (0)
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  return b;
}

BytesStatic bytes_static_truncate(const BytesStatic *src, size_t new_len)
{
__ESBMC_HIDE:;
  /* Constant-index unroll — see `bytes_static_from_string` rationale. */
  BytesStatic b = {0};
  b.length = new_len;
#define STEP(i)                                                                \
  do                                                                           \
  {                                                                            \
    if ((i) < new_len)                                                         \
      b.data[i] = src->data[i];                                                \
  } while (0)
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  return b;
}

BytesStatic bytes_static_and(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
#define STEP(i)                                                                \
  if (a->length > (i))                                                         \
  r.data[i] = a->data[i] & b->data[i]
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  r.length = a->length;
  return r;
}

BytesStatic bytes_static_or(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
#define STEP(i)                                                                \
  if (a->length > (i))                                                         \
  r.data[i] = a->data[i] | b->data[i]
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  r.length = a->length;
  return r;
}

BytesStatic bytes_static_xor(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
#define STEP(i)                                                                \
  if (a->length > (i))                                                         \
  r.data[i] = a->data[i] ^ b->data[i]
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  r.length = a->length;
  return r;
}

/* Big-endian decode of a BytesStatic into a uint256, valid for any
 * `b->length` in [0, 32].  Each STEP(i) uses a literal index so
 * b->data[i] is a constant memory offset; the conditional preserves
 * the accumulator past length, exactly equivalent to the original
 * `for (i < length)` loop.  No symbolic-bound loop, so symex never
 * has to unwind and stderr stays clean even when the caller's
 * `--unwind` is below 32 (which used to truncate the result and
 * flood the log with "Not unwinding" lines per visit). */
uint256_t bytes_static_to_uint(const BytesStatic *b)
{
__ESBMC_HIDE:;
  uint256_t r = 0;
#define STEP(i) r = (b->length > (i)) ? ((r << 8) | b->data[i]) : r
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  return r;
}

/* Big-endian encode of a uint256 into a BytesStatic, valid for any
 * `len` in [0, 32].  Symmetric with bytes_static_to_uint: STEP(j)
 * uses literal index j and writes b.data[j] = (val >> (8 * (len-1-j)))
 * & 0xFF when j < len.  The shift amount is symbolic but symex
 * handles symbolic-shift bitvector ops directly without unwinding. */
BytesStatic bytes_static_from_uint(uint256_t val, size_t len)
{
__ESBMC_HIDE:;
  BytesStatic b = {0};
#define STEP(j)                                                                \
  if (len > (j))                                                               \
  b.data[j] = (unsigned char)(val >> (8 * (len - 1 - (j))))
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  b.length = len;
  return b;
}

BytesStatic bytes_static_shl(const BytesStatic *src, unsigned shift_bits)
{
__ESBMC_HIDE:;
  uint256_t val = bytes_static_to_uint(src);
  val <<= shift_bits;
  return bytes_static_from_uint(val, src->length);
}

BytesStatic bytes_static_shr(const BytesStatic *src, unsigned shift_bits)
{
__ESBMC_HIDE:;
  uint256_t val = bytes_static_to_uint(src);
  val >>= shift_bits;
  return bytes_static_from_uint(val, src->length);
}

uint256_t bytes_static_to_mapping_key(const BytesStatic *b)
{
__ESBMC_HIDE:;
  return ((uint256_t)b->length << 248) | bytes_static_to_uint(b);
}

BytesStatic bytes_static_init_zero(size_t len)
{
__ESBMC_HIDE:;
  /* `BytesStatic b = {0}` already zero-inits the 32-byte data array
   * at compile time; no runtime memset needed. The prior memset(b.data,
   * 0, len) was redundant AND triggered __memset_impl unrolling
   * 1..len times on every visit. */
  BytesStatic b = {0};
  b.length = len;
  return b;
}

BytesDynamic bytes_dynamic_init_zero(size_t len, BytesPool *pool)
{
__ESBMC_HIDE:;
  BytesDynamic b = {0};
  b.offset = pool->pool_cursor;
  b.length = len;
  b.capacity = len;
  b.initialized = 1;
  memset(&pool->pool[b.offset], 0, len);
  pool->pool_cursor += len;
  return b;
}

void bytes_dynamic_init(
  BytesDynamic *b,
  const unsigned char *input,
  size_t len,
  BytesPool *pool)
{
__ESBMC_HIDE:;
  b->offset = pool->pool_cursor;
  b->length = len;
  b->capacity = len;
  b->initialized = 1;
  memcpy(&pool->pool[b->offset], input, len);
  pool->pool_cursor += len;
}

void bytes_dynamic_ensure_capacity(
  BytesDynamic *b,
  size_t required,
  BytesPool *pool)
{
__ESBMC_HIDE:;
  if (required <= b->capacity)
    return;
  size_t new_capacity = b->capacity;
  if (new_capacity == 0)
    new_capacity = 1;
  while (new_capacity < required)
    new_capacity *= 2;
  size_t new_offset = pool->pool_cursor;
  memcpy(&pool->pool[new_offset], &pool->pool[b->offset], b->length);
  b->offset = new_offset;
  b->capacity = new_capacity;
  pool->pool_cursor += new_capacity;
}

BytesDynamic bytes_dynamic_from_static(const BytesStatic *s, BytesPool *pool)
{
__ESBMC_HIDE:;
  BytesDynamic b = {0};
  bytes_dynamic_init(&b, s->data, s->length, pool);
  return b;
}

BytesDynamic bytes_dynamic_from_string(const char *str, BytesPool *pool)
{
__ESBMC_HIDE:;
  /* NULL str = Solidity's default empty string (`""`), length 0.
   * Mapping/struct reads of unset `string`-typed fields lower to NULL
   * char* in our IR; without this guard, strlen(NULL) is UB and
   * `--no-standard-checks` lets symex pick a nondet length, breaking
   * `bytes(default_string).length == 0`. Pinned by `bytes_string_1`. */
  BytesDynamic b = {0};
  if (str == 0)
  {
    b.initialized = 1;
    return b;
  }
  bytes_dynamic_init(&b, (const unsigned char *)str, strlen(str), pool);
  return b;
}

BytesDynamic
bytes_dynamic_from_hex(const char *hex_str, size_t len, BytesPool *pool)
{
__ESBMC_HIDE:;
  size_t hex_len = len - 2;
  size_t byte_len = hex_len / 2;
  unsigned char tmp[32] = {0};
  for (size_t i = 0; i < byte_len; i++)
  {
    unsigned char high = hex_char_to_nibble(hex_str[2 + i * 2]);
    unsigned char low = hex_char_to_nibble(hex_str[2 + i * 2 + 1]);
    tmp[i] = (high << 4) | low;
  }
  BytesDynamic b = {0};
  bytes_dynamic_init(&b, tmp, byte_len, pool);
  return b;
}

BytesStatic bytes_static_truncate_from_dynamic(
  const BytesDynamic *src,
  size_t new_len,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  BytesStatic b = {0};
  memcpy(b.data, &pool->pool[src->offset], new_len);
  b.length = new_len;
  return b;
}

BytesDynamic
bytes_dynamic_concat(BytesDynamic a, BytesDynamic b, BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(a.initialized);
  bytes_dynamic_init_check(b.initialized);
  BytesDynamic d = {0};
  d.offset = pool->pool_cursor;
  d.length = a.length + b.length;
  d.capacity = d.length;
  d.initialized = 1;
  memcpy(&pool->pool[d.offset], &pool->pool[a.offset], a.length);
  memcpy(&pool->pool[d.offset + a.length], &pool->pool[b.offset], b.length);
  pool->pool_cursor += d.length;
  return d;
}

BytesDynamic bytes_dynamic_copy(const BytesDynamic *src, BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  BytesDynamic d = {0};
  d.offset = pool->pool_cursor;
  d.length = src->length;
  d.capacity = src->length;
  d.initialized = 1;
  memcpy(&pool->pool[d.offset], &pool->pool[src->offset], src->length);
  pool->pool_cursor += d.length;
  return d;
}

void bytes_static_set(BytesStatic *b, size_t index, BytesStatic value)
{
__ESBMC_HIDE:;
  b->data[index] = value.data[0];
}

void bytes_dynamic_set(
  BytesDynamic *b,
  size_t index,
  BytesStatic value,
  BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  bytes_dynamic_ensure_capacity(b, index + 1, pool);
  pool->pool[b->offset + index] = value.data[0];
  if (index >= b->length)
  {
    b->length = index + 1;
  }
}

BytesStatic bytes_static_get(const BytesStatic *b, size_t index)
{
__ESBMC_HIDE:;
  BytesStatic r = {0};
  r.data[0] = b->data[index];
  r.length = 1;
  return r;
}

BytesStatic
bytes_dynamic_get(const BytesDynamic *b, const BytesPool *pool, size_t index)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  bytes_dynamic_bounds_check(index, b->length);
  BytesStatic r = {0};
  r.data[0] = pool->pool[b->offset + index];
  r.length = 1;
  return r;
}

bool bytes_static_equal(const BytesStatic *a, const BytesStatic *b)
{
__ESBMC_HIDE:;
  if (a->length != b->length)
    return false;
  // Straight-line comparison — do not call memcmp here. string.c's
  // memcmp is a byte-stepping loop which k-induction bounds to k
  // iterations, silently truncating the comparison for bytes beyond
  // the k-th position and producing false-positive equality. bytesN
  // (1 <= N <= 32) has a compile-time-bounded data buffer, so unroll
  // 32 length-gated byte comparisons explicitly. ESBMC emits one
  // conjunction of equalities per call, no loop unwinding needed.
  const size_t n = a->length;
#define __ESBMC_BSEQ(i) (n <= (size_t)(i) || a->data[i] == b->data[i])
  return __ESBMC_BSEQ(0) && __ESBMC_BSEQ(1) && __ESBMC_BSEQ(2) &&
         __ESBMC_BSEQ(3) && __ESBMC_BSEQ(4) && __ESBMC_BSEQ(5) &&
         __ESBMC_BSEQ(6) && __ESBMC_BSEQ(7) && __ESBMC_BSEQ(8) &&
         __ESBMC_BSEQ(9) && __ESBMC_BSEQ(10) && __ESBMC_BSEQ(11) &&
         __ESBMC_BSEQ(12) && __ESBMC_BSEQ(13) && __ESBMC_BSEQ(14) &&
         __ESBMC_BSEQ(15) && __ESBMC_BSEQ(16) && __ESBMC_BSEQ(17) &&
         __ESBMC_BSEQ(18) && __ESBMC_BSEQ(19) && __ESBMC_BSEQ(20) &&
         __ESBMC_BSEQ(21) && __ESBMC_BSEQ(22) && __ESBMC_BSEQ(23) &&
         __ESBMC_BSEQ(24) && __ESBMC_BSEQ(25) && __ESBMC_BSEQ(26) &&
         __ESBMC_BSEQ(27) && __ESBMC_BSEQ(28) && __ESBMC_BSEQ(29) &&
         __ESBMC_BSEQ(30) && __ESBMC_BSEQ(31);
#undef __ESBMC_BSEQ
}

bool bytes_dynamic_equal(
  const BytesDynamic *a,
  const BytesDynamic *b,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(a->initialized);
  bytes_dynamic_init_check(b->initialized);
  if (a->length != b->length)
    return false;
  return memcmp(&pool->pool[a->offset], &pool->pool[b->offset], a->length) == 0;
}

uint256_t
bytes_dynamic_to_mapping_key(const BytesDynamic *b, const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  uint256_t result = 0;
  for (size_t i = 0; i < b->length; i++)
  {
    result = (result << 8) | pool->pool[b->offset + i];
  }
  result |= ((uint256_t)b->length) << 248;
  return result;
}

void bytes_dynamic_push(BytesDynamic *b, unsigned char value, BytesPool *pool)
{
__ESBMC_HIDE:;
  if (!b->initialized)
  {
    b->offset = pool->pool_cursor;
    b->length = 0;
    b->capacity = 4;
    b->initialized = 1;
    pool->pool_cursor += b->capacity;
  }
  bytes_dynamic_ensure_capacity(b, b->length + 1, pool);
  pool->pool[b->offset + b->length] = value;
  b->length++;
}

void bytes_dynamic_pop(BytesDynamic *b, BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  bytes_dynamic_bounds_check(0, b->length);
  b->length--;
}

uint256_t bytes_dynamic_to_uint(const BytesDynamic *b, const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  uint256_t result = 0;
  for (size_t i = 0; i < b->length; i++)
  {
    result = (result << 8) | pool->pool[b->offset + i];
  }
  return result;
}

char *bytes_static_to_string(const BytesStatic *b)
{
__ESBMC_HIDE:;
  char *out = (char *)malloc(b->length + 1);
  for (size_t i = 0; i < b->length; i++)
  {
    out[i] = (char)b->data[i];
  }
  out[b->length] = '\0';
  return out;
}

char *bytes_dynamic_to_string(const BytesDynamic *b, const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(b->initialized);
  char *out = (char *)malloc(b->length + 1);
  for (size_t i = 0; i < b->length; i++)
  {
    out[i] = (char)pool->pool[b->offset + i];
  }
  out[b->length] = '\0';
  return out;
}

BytesStatic bytes_static_extend(const BytesStatic *src, size_t new_len)
{
__ESBMC_HIDE:;
  /* Constant-index unroll — see `bytes_static_from_string` rationale.
   * Zero-pad above src->length is implicit via the `{0}` initializer. */
  BytesStatic out = {0};
  out.length = new_len;
#define STEP(i)                                                                \
  do                                                                           \
  {                                                                            \
    if ((i) < src->length)                                                     \
      out.data[i] = src->data[i];                                              \
  } while (0)
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  return out;
}

BytesStatic bytes_static_resize(const BytesStatic *src, size_t new_len)
{
__ESBMC_HIDE:;
  if (new_len == src->length)
  {
    return *src;
  }
  else if (new_len < src->length)
  {
    return bytes_static_truncate(src, new_len);
  }
  else
  {
    return bytes_static_extend(src, new_len);
  }
}

BytesStatic bytes_static_extend_from_dynamic(
  const BytesDynamic *src,
  size_t new_len,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  /* Constant-index unroll over the BytesDynamic source bytes. */
  BytesStatic b = {0};
  b.length = new_len;
#define STEP(i)                                                                \
  do                                                                           \
  {                                                                            \
    if ((i) < src->length)                                                     \
      b.data[i] = pool->pool[src->offset + (i)];                               \
  } while (0)
  _ESBMC_BS_UNROLL_32(STEP);
#undef STEP
  return b;
}

BytesStatic bytes_static_resize_from_dynamic(
  const BytesDynamic *src,
  size_t new_len,
  const BytesPool *pool)
{
__ESBMC_HIDE:;
  bytes_dynamic_init_check(src->initialized);
  if (new_len == src->length)
  {
    BytesStatic b = {0};
    memcpy(b.data, &pool->pool[src->offset], new_len);
    b.length = new_len;
    return b;
  }
  else if (new_len < src->length)
  {
    return bytes_static_truncate_from_dynamic(src, new_len, pool);
  }
  else
  {
    return bytes_static_extend_from_dynamic(src, new_len, pool);
  }
}

BytesPool bytes_pool_init(unsigned char *pool_data)
{
__ESBMC_HIDE:;
  BytesPool pool = {pool_data, 0};
  return pool;
}
