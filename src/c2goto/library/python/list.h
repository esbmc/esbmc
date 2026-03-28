#ifndef PYTHON_LIST_H
#define PYTHON_LIST_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "python_types.h"

/**
 * @brief Compare two value buffers for equality.
 *
 * Optimized for the sizes the Python frontend commonly emits:
 *   8 bytes  – int / float
 *   1 byte   – bool
 * Falls back to memcmp for any other size.
 */
static inline bool
__ESBMC_values_equal(const void *a, const void *b, size_t size)
{
  if (a == b)
    return true;
  if (size == 8)
    return *(const uint64_t *)a == *(const uint64_t *)b;
  if (size == 1)
    return *(const uint8_t *)a == *(const uint8_t *)b;
  return memcmp(a, b, size) == 0;
}

/* ------------------------------------------------------------------ */
/* Public list API                                                      */
/* ------------------------------------------------------------------ */

PyListObject *__ESBMC_list_create(void);
size_t __ESBMC_list_size(const PyListObject *l);

bool __ESBMC_list_push(
  PyListObject *l,
  const void *value,
  size_t type_id,
  size_t type_size);

bool __ESBMC_list_push_object(PyListObject *l, PyObject *o);
bool __ESBMC_list_push_dict_ptr(
  PyListObject *l,
  void *dict_ptr,
  size_t type_id);

bool __ESBMC_list_eq(
  const PyListObject *l1,
  const PyListObject *l2,
  size_t list_type_id,
  size_t max_depth);

bool __ESBMC_list_set_eq(const PyListObject *l1, const PyListObject *l2);

PyObject *__ESBMC_list_at(PyListObject *l, size_t index);

bool __ESBMC_list_set_at(
  PyListObject *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size);

bool __ESBMC_list_insert(
  PyListObject *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size);

bool __ESBMC_list_contains(
  const PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size);

void __ESBMC_list_extend(PyListObject *l, const PyListObject *other);
void __ESBMC_list_clear(PyListObject *l);

size_t __ESBMC_list_find_index(
  PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size);

size_t __ESBMC_list_try_find_index(
  PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size);

bool __ESBMC_list_remove_at(PyListObject *l, size_t index);
PyObject *__ESBMC_list_pop(PyListObject *l, int64_t index);
PyListObject *__ESBMC_list_copy(const PyListObject *l);

bool __ESBMC_list_remove(
  PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size);

void __ESBMC_list_sort(PyListObject *l, int type_flag, uint64_t float_type_id);
void __ESBMC_list_reverse(PyListObject *l);

#endif /* PYTHON_LIST_H */
