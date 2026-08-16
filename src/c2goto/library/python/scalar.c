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
  size_t type_id,
  long long value)
{
__ESBMC_HIDE:;
  if (!tagged || tagged->type_id != type_id)
    return 0;
  return *(const long long *)tagged->value == value;
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

// Models a runtime type mismatch (e.g. `x + 1` where `x` holds a str) as a
// Python TypeError, the same way IndexError/KeyError are modeled elsewhere
// in this library: an assert on the path that would have raised.
long long
__python_scalar_add_num(const PyObject *tagged, size_t type_id, long long value)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && tagged->type_id == type_id,
    "TypeError: unsupported operand type(s) for +");
  if (!tagged || tagged->type_id != type_id)
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
  size_t type_id,
  long long value,
  int tagged_is_left)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && tagged->type_id == type_id,
    "TypeError: unsupported operand type(s) for -");
  if (!tagged || tagged->type_id != type_id)
    return 0;
  long long tagged_val = *(const long long *)tagged->value;
  return tagged_is_left ? tagged_val - value : value - tagged_val;
}

// Python's / is always true division, even for two ints.
double __python_scalar_div_num(
  const PyObject *tagged,
  size_t type_id,
  long long value,
  int tagged_is_left)
{
__ESBMC_HIDE:;
  __ESBMC_assert(
    tagged && tagged->type_id == type_id,
    "TypeError: unsupported operand type(s) for /");
  if (!tagged || tagged->type_id != type_id)
    return 0.0;
  long long tagged_val = *(const long long *)tagged->value;
  return tagged_is_left ? (double)tagged_val / (double)value
                        : (double)value / (double)tagged_val;
}
