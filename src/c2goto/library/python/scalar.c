#include <stdint.h>
#include <string.h>
#include "python_types.h"

// Byte-compare with fast paths for the sizes short scalar strings actually
// use; falls back to memcmp otherwise.
static inline int
__python_scalar_bytes_equal(const void *a, const void *b, size_t size)
{
__ESBMC_HIDE:;
  if (size == 1)
    return *(const uint8_t *)a == *(const uint8_t *)b;
  if (size == 2)
    return *(const uint16_t *)a == *(const uint16_t *)b;
  if (size == 4)
    return *(const uint32_t *)a == *(const uint32_t *)b;
  if (size == 8)
    return *(const uint64_t *)a == *(const uint64_t *)b;
  return memcmp(a, b, size) == 0;
}

// Copy with fast paths for the sizes short scalar strings/values actually use.
static inline void
__python_scalar_bytes_copy(void *dst, const void *src, size_t size)
{
__ESBMC_HIDE:;
  if (size == 1)
    *(uint8_t *)dst = *(const uint8_t *)src;
  else if (size == 2)
    *(uint16_t *)dst = *(const uint16_t *)src;
  else if (size == 4)
    *(uint32_t *)dst = *(const uint32_t *)src;
  else if (size == 8)
    *(uint64_t *)dst = *(const uint64_t *)src;
  else
    memcpy(dst, src, size);
}

// Copies `size` bytes from `value` into non-expiring storage, so a
// PyObject's `.value` pointer stays valid after the assigning branch's
// locals go DEAD at the join.
void *__python_scalar_tag_copy(const void *value, size_t size)
{
__ESBMC_HIDE:;
  void *copied = __ESBMC_alloca(size);
  __python_scalar_bytes_copy(copied, value, size);
  return copied;
}

int __python_scalar_eq_num(
  const PyObject *tagged,
  int type_matches,
  long long value)
{
__ESBMC_HIDE:;
  if (!tagged || !type_matches)
    return 0;
  return *(const long long *)tagged->value == value;
}

// Tagged-vs-tagged equality. Dispatching on `type_id` before any byte compare
// keeps the numeric arm at a fixed width, so only two tagged strings reach
// the byte loop below, which the numeric case would otherwise pay for too.
int __python_scalar_eq_obj(
  const PyObject *a,
  const PyObject *b,
  size_t num_type_id)
{
__ESBMC_HIDE:;
  // Python compares across types as unequal rather than coercing.
  if (a->type_id != b->type_id)
    return 0;
  if (a->type_id == num_type_id)
    return *(const long long *)a->value == *(const long long *)b->value;
  if (a->size != b->size)
    return 0;
  // `.size` is only known at runtime, so memcmp's loop bound would be
  // symbolic and never unwind to completion; a compile-time bound keeps this
  // loop finite. Do not "simplify" it back to memcmp. `.size` counts the
  // trailing NUL, so the limit here is 255 characters -- one below the length
  // bound __python_strnlen_bounded applies.
  __ESBMC_assert(
    a->size <= ESBMC_PY_STRNLEN_BOUND, "tagged str exceeds the modelled bound");
  for (size_t i = 0; i < ESBMC_PY_STRNLEN_BOUND; ++i)
  {
    if (i >= a->size)
      break;
    if (((const uint8_t *)a->value)[i] != ((const uint8_t *)b->value)[i])
      return 0;
  }
  return 1;
}

int __python_scalar_eq_str(
  const PyObject *tagged,
  size_t type_id,
  const char *value,
  size_t size)
{
__ESBMC_HIDE:;
  // False on a type_id or size mismatch.
  if (!tagged || tagged->type_id != type_id || tagged->size != size)
    return 0;
  return __python_scalar_bytes_equal(tagged->value, value, size);
}

// Three-way compare against a numeric literal (-1/0/1). Shared by
// Lt/LtE/Gt/GtE -- the frontend composes the boolean from this against 0.
int __python_scalar_cmp_num(
  const PyObject *tagged,
  int type_matches,
  long long value)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && type_matches,
    "TypeError: comparison not supported between these types");
  if (!tagged || !type_matches)
    return 0;
  long long tagged_val = *(const long long *)tagged->value;
  if (tagged_val < value)
    return -1;
  if (tagged_val > value)
    return 1;
  return 0;
}

// Lexicographic three-way compare against a literal str (-1/0/1). The loop
// bound (`value_size`) is the literal's own compile-time length, not a
// symbolic memcmp-style n. `.size`/`value_size` include the trailing '\0',
// so the `i >= tagged->size` guard (stopping the read there) already gives
// the correct prefix ordering ("ab" < "abc") for free.
int __python_scalar_cmp_str(
  const PyObject *tagged,
  size_t type_id,
  const char *value,
  size_t value_size)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && tagged->type_id == type_id,
    "TypeError: comparison not supported between these types");
  if (!tagged || tagged->type_id != type_id)
    return 0;

  for (size_t i = 0; i < value_size; ++i)
  {
    if (i >= tagged->size)
      return -1;
    unsigned char a = ((const unsigned char *)tagged->value)[i];
    unsigned char b = (unsigned char)value[i];
    if (a != b)
      return a < b ? -1 : 1;
  }
  return tagged->size > value_size ? 1 : 0;
}

// Models a runtime type mismatch (e.g. `x + 1` where `x` holds a str) as a
// Python TypeError, the same way IndexError/KeyError are modeled elsewhere
// in this library: an assert on the path that would have raised.
long long __python_scalar_add_num(
  const PyObject *tagged,
  int type_matches,
  long long value)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && type_matches, "TypeError: unsupported operand type(s) for +");
  if (!tagged || !type_matches)
    return 0;
  return *(const long long *)tagged->value + value;
}

// Concatenation order matters, so `tagged_is_left` says which side of `+`
// `tagged` was on. `size` and `value_size` include each operand's trailing
// '\0' -- stripped here so the result ends up with exactly one, at the end.
char *__python_scalar_add_str(
  const PyObject *tagged,
  size_t type_id,
  const char *value,
  size_t value_size,
  int tagged_is_left)
{
__ESBMC_HIDE:;
  int type_matches = tagged && tagged->type_id == type_id;
  __ESBMC_assert(
    type_matches,
    "TypeError: can only concatenate str (not other types) to str");

  size_t tagged_len = tagged->size - 1;
  size_t value_len = value_size - 1;
  char *buffer = __ESBMC_alloca(tagged_len + value_len + 1);
  char *tagged_dst = tagged_is_left ? buffer : buffer + value_len;
  char *value_dst = tagged_is_left ? buffer + tagged_len : buffer;

  // Zero-fill the tagged side on a mismatch instead of skipping it, so
  // the buffer keeps the same shape either way.
  if (type_matches)
    __python_scalar_bytes_copy(tagged_dst, tagged->value, tagged_len);
  else
    memset(tagged_dst, 0, tagged_len);
  __python_scalar_bytes_copy(value_dst, value, value_len);

  buffer[tagged_len + value_len] = '\0';
  return buffer;
}

// `tagged_is_left` records which side of `-` `tagged` was on, since
// subtraction order matters.
long long __python_scalar_sub_num(
  const PyObject *tagged,
  int type_matches,
  long long value,
  int tagged_is_left)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && type_matches, "TypeError: unsupported operand type(s) for -");
  if (!tagged || !type_matches)
    return 0;
  long long tagged_val = *(const long long *)tagged->value;
  return tagged_is_left ? tagged_val - value : value - tagged_val;
}

// Python's / is always true division, even for two ints.
double __python_scalar_div_num(
  const PyObject *tagged,
  int type_matches,
  long long value,
  int tagged_is_left)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && type_matches, "TypeError: unsupported operand type(s) for /");
  if (!tagged || !type_matches)
    return 0.0;
  long long tagged_val = *(const long long *)tagged->value;
  return tagged_is_left ? (double)tagged_val / (double)value
                        : (double)value / (double)tagged_val;
}

// Both operands tagged, so lhs/rhs already match the AST order.
long long __python_scalar_sub_num_dyn(
  const PyObject *lhs,
  int lhs_is_num,
  const PyObject *rhs,
  int rhs_is_num)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    lhs && rhs && lhs_is_num && rhs_is_num,
    "TypeError: unsupported operand type(s) for -");
  if (!lhs || !rhs || !lhs_is_num || !rhs_is_num)
    return 0;
  return *(const long long *)lhs->value - *(const long long *)rhs->value;
}

double __python_scalar_div_num_dyn(
  const PyObject *lhs,
  int lhs_is_num,
  const PyObject *rhs,
  int rhs_is_num)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    lhs && rhs && lhs_is_num && rhs_is_num,
    "TypeError: unsupported operand type(s) for /");
  if (!lhs || !rhs || !lhs_is_num || !rhs_is_num)
    return 0.0;
  return (double)(*(const long long *)lhs->value) /
         (double)(*(const long long *)rhs->value);
}

// Both operands tagged, so the result's own type is only known at runtime --
// it comes back as a tagged object, not a raw scalar. The str buffer's
// __ESBMC_alloca storage survives past this call, like
// __python_scalar_tag_copy.
PyObject __python_scalar_add_obj_dyn(
  const PyObject *lhs,
  int lhs_is_num,
  int lhs_is_str,
  const PyObject *rhs,
  int rhs_is_num,
  int rhs_is_str,
  size_t num_type_id,
  size_t str_type_id)
{
__ESBMC_HIDE:;
  int both_num = lhs && rhs && lhs_is_num && rhs_is_num;
  int both_str = lhs && rhs && lhs_is_str && rhs_is_str;
  __ESBMC_assert(
    both_num || both_str, "TypeError: unsupported operand type(s) for +");

  PyObject result;
  result.float_idx = 0;

  if (both_num)
  {
    long long *sum = __ESBMC_alloca(sizeof(long long));
    *sum = *(const long long *)lhs->value + *(const long long *)rhs->value;
    result.value = sum;
    result.type_id = num_type_id;
    result.size = sizeof(long long);
    return result;
  }

  // both_str. `.size` includes the trailing '\0' on each side, stripped so
  // the result ends up with exactly one, at the end. Lengths are only known
  // at runtime, so the copy loops below use a compile-time bound instead of
  // memcpy, whose loop bound would be symbolic and never unwind fully.
  size_t lhs_len = both_str ? lhs->size - 1 : 0;
  size_t rhs_len = both_str ? rhs->size - 1 : 0;
  __ESBMC_assert(
    lhs_len <= ESBMC_PY_STRNLEN_BOUND && rhs_len <= ESBMC_PY_STRNLEN_BOUND,
    "tagged str exceeds the modelled bound");

  char *buffer = __ESBMC_alloca(lhs_len + rhs_len + 1);
  for (size_t i = 0; i < ESBMC_PY_STRNLEN_BOUND; ++i)
  {
    if (i >= lhs_len)
      break;
    buffer[i] = ((const char *)lhs->value)[i];
  }
  for (size_t i = 0; i < ESBMC_PY_STRNLEN_BOUND; ++i)
  {
    if (i >= rhs_len)
      break;
    buffer[lhs_len + i] = ((const char *)rhs->value)[i];
  }
  buffer[lhs_len + rhs_len] = '\0';

  result.value = buffer;
  result.type_id = str_type_id;
  result.size = lhs_len + rhs_len + 1;
  return result;
}
