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

// Copies `size` bytes from `value` into non-expiring storage, so a
// PyObject's `.value` pointer stays valid after the assigning branch's
// locals go DEAD at the join.
void *__python_scalar_tag_copy(const void *value, size_t size)
{
__ESBMC_HIDE:;
  void *copied = __ESBMC_alloca(size);
  if (size == 1)
    *(uint8_t *)copied = *(const uint8_t *)value;
  else if (size == 2)
    *(uint16_t *)copied = *(const uint16_t *)value;
  else if (size == 4)
    *(uint32_t *)copied = *(const uint32_t *)value;
  else if (size == 8)
    *(uint64_t *)copied = *(const uint64_t *)value;
  else
    memcpy(copied, value, size);
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
