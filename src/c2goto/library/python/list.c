#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>
#include "python_types.h"

// TODO: There is no such a thing as a generic type in python.
static PyType __ESBMC_generic_type;
static PyType __ESBMC_list_type;

// Optimized value comparison - avoids memcmp loop unrolling for common sizes
static inline bool
__ESBMC_values_equal(const void *a, const void *b, size_t size)
{
  if (a == b)
    return true;
  // Direct comparison for common sizes - no loop needed
  // Python frontend maps: int/float -> 8 bytes, bool -> 1 byte
  if (size == 8)
    return *(const uint64_t *)a == *(const uint64_t *)b;
  if (size == 4)
    return *(const uint32_t *)a == *(const uint32_t *)b;
  if (size == 2)
    return *(const uint16_t *)a == *(const uint16_t *)b;
  if (size == 1)
    return *(const uint8_t *)a == *(const uint8_t *)b;
  if (size == 16)
  {
    return *(const uint64_t *)a == *(const uint64_t *)b &&
           *((const uint64_t *)a + 1) == *((const uint64_t *)b + 1);
  }
  // Fallback for larger/unusual sizes
  return memcmp(a, b, size) == 0;
}

// Default maximum nesting depth to prevent state explosion during symbolic execution.
// This can be overridden via --python-list-compare-depth option.
#define __ESBMC_LIST_DEFAULT_DEPTH 4

// Maximum physical stack size for list comparison (prevents buffer overflow).
// Set to 64 to allow users to increase depth without hitting buffer limits.
#define __ESBMC_LIST_MAX_STACK 64

static inline const char *
__ESBMC_list_elem_to_str(const PyObject *obj)
{
  if (!obj || !obj->value)
    return NULL;

  if (obj->size == sizeof(char *))
    return *(const char *const *)obj->value;

  return (const char *)obj->value;
}

static inline const char *
__ESBMC_list_item_to_str(const void *item, size_t item_size)
{
  if (!item)
    return NULL;

  if (item_size == sizeof(char *))
    return *(const char *const *)item;

  return (const char *)item;
}

static inline size_t
__ESBMC_strnlen_bounded(const char *s, size_t max_len)
{
  if (!s)
    return 0;
  if (max_len == 0)
    return 0;
  // Unrolled up to 16 to avoid loops in BMC.
  if (s[0] == '\0' || max_len <= 0) return 0;
  if (s[1] == '\0' || max_len <= 1) return 1;
  if (s[2] == '\0' || max_len <= 2) return 2;
  if (s[3] == '\0' || max_len <= 3) return 3;
  if (s[4] == '\0' || max_len <= 4) return 4;
  if (s[5] == '\0' || max_len <= 5) return 5;
  if (s[6] == '\0' || max_len <= 6) return 6;
  if (s[7] == '\0' || max_len <= 7) return 7;
  if (s[8] == '\0' || max_len <= 8) return 8;
  if (s[9] == '\0' || max_len <= 9) return 9;
  if (s[10] == '\0' || max_len <= 10) return 10;
  if (s[11] == '\0' || max_len <= 11) return 11;
  if (s[12] == '\0' || max_len <= 12) return 12;
  if (s[13] == '\0' || max_len <= 13) return 13;
  if (s[14] == '\0' || max_len <= 14) return 14;
  if (s[15] == '\0' || max_len <= 15) return 15;
  return (max_len < 16) ? max_len : 16;
}

static inline bool
__ESBMC_str_equal_bounded(const char *a, const char *b, size_t max_len)
{
  if (a == b)
    return true;
  if (!a || !b)
    return false;
  size_t len_a = __ESBMC_strnlen_bounded(a, max_len);
  size_t len_b = __ESBMC_strnlen_bounded(b, max_len);
  if (len_a != len_b)
    return false;
  // Unrolled compare up to 16 chars
  if (len_a == 0) return true;
  if (a[0] != b[0]) return false;
  if (len_a == 1) return true;
  if (a[1] != b[1]) return false;
  if (len_a == 2) return true;
  if (a[2] != b[2]) return false;
  if (len_a == 3) return true;
  if (a[3] != b[3]) return false;
  if (len_a == 4) return true;
  if (a[4] != b[4]) return false;
  if (len_a == 5) return true;
  if (a[5] != b[5]) return false;
  if (len_a == 6) return true;
  if (a[6] != b[6]) return false;
  if (len_a == 7) return true;
  if (a[7] != b[7]) return false;
  if (len_a == 8) return true;
  if (a[8] != b[8]) return false;
  if (len_a == 9) return true;
  if (a[9] != b[9]) return false;
  if (len_a == 10) return true;
  if (a[10] != b[10]) return false;
  if (len_a == 11) return true;
  if (a[11] != b[11]) return false;
  if (len_a == 12) return true;
  if (a[12] != b[12]) return false;
  if (len_a == 13) return true;
  if (a[13] != b[13]) return false;
  if (len_a == 14) return true;
  if (a[14] != b[14]) return false;
  if (len_a == 15) return true;
  if (a[15] != b[15]) return false;
  return true;
}

static inline bool __ESBMC_list_elem_eq_simple(
  const PyObject *a,
  const PyObject *b,
  size_t list_type_id,
  size_t string_type_id)
{
  if (a->value == b->value)
    return true;

  if (a->type_id == string_type_id && b->type_id == string_type_id)
  {
    const char *sa = __ESBMC_list_elem_to_str(a);
    const char *sb = __ESBMC_list_elem_to_str(b);
    if (!sa || !sb)
      return false;
    return __ESBMC_str_equal_bounded(sa, sb, 64);
  }

  if (!a->value || !b->value)
    return false;
  if (a->type_id != b->type_id)
    return false;
  if (a->size != b->size)
    return false;

  // Defer nested list comparison to slow path
  if (a->type_id == list_type_id)
    return false;

  return __ESBMC_values_equal(a->value, b->value, a->size);
}

PyObject *__ESBMC_create_inf_obj()
{
  return NULL;
};

PyListObject *__ESBMC_list_create()
{
  PyListObject *l = __ESBMC_alloca(sizeof(PyListObject));
  l->type = &__ESBMC_list_type;
  l->items = __ESBMC_create_inf_obj();
  l->size = 0;
  return l;
}

size_t __ESBMC_list_size(const PyListObject *l)
{
  return l ? l->size : 0;
}

static inline void *__ESBMC_copy_value(const void *value, size_t size)
{
  // None type (NULL pointer with size 0)
  // Don't allocate: return NULL to preserve None semantics
  if (value == NULL && size == 0)
    return NULL;

  void *copied = __ESBMC_alloca(size);

  if (size == 8)
    *(uint64_t *)copied = *(const uint64_t *)value;
  else if (size == 4)
    *(uint32_t *)copied = *(const uint32_t *)value;
  else if (size == 2)
    *(uint16_t *)copied = *(const uint16_t *)value;
  else if (size == 16)
  {
    // Handle 16-byte structs (such as dictionaries) explicitly
    *(uint64_t *)copied = *(const uint64_t *)value;
    *((uint64_t *)copied + 1) = *((const uint64_t *)value + 1);
  }
  else
    memcpy(copied, value, size);

  return copied;
}

bool __ESBMC_list_push(
  PyListObject *l,
  const void *value,
  size_t type_id,
  size_t type_size)
{
  // TODO: __ESBMC_obj_cpy
  void *copied_value = __ESBMC_copy_value(value, type_size);

  // Use a pointer to avoid repeated indexing
  PyObject *item = &l->items[l->size];
  item->value = copied_value;
  item->type_id = type_id;
  item->size = type_size;
  l->size++;

  // TODO: Nondeterministic failure?
  return true;
}

bool __ESBMC_list_push_object(PyListObject *l, PyObject *o)
{
  assert(l != NULL);
  assert(o != NULL);
  return __ESBMC_list_push(l, o->value, o->type_id, o->size);
}

bool __ESBMC_list_eq(
  const PyListObject *l1,
  const PyListObject *l2,
  size_t list_type_id,
  size_t string_type_id,
  size_t max_depth)
{
  // Quick checks
  if (!l1 || !l2)
    return false;
  if (__ESBMC_same_object(l1, l2))
    return true;
  if (l1->size != l2->size)
    return false;

  // Fast path for small, non-nested lists (avoids loops in the model).
  const size_t small_max = 8;
  __ESBMC_assume(l1->size <= small_max);
  if (l1->size <= small_max)
  {
    if (l1->size > 0)
    {
      const PyObject *a0 = &l1->items[0];
      const PyObject *b0 = &l2->items[0];
      if (a0->type_id == list_type_id || b0->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a0, b0, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 1)
    {
      const PyObject *a1 = &l1->items[1];
      const PyObject *b1 = &l2->items[1];
      if (a1->type_id == list_type_id || b1->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a1, b1, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 2)
    {
      const PyObject *a2 = &l1->items[2];
      const PyObject *b2 = &l2->items[2];
      if (a2->type_id == list_type_id || b2->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a2, b2, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 3)
    {
      const PyObject *a3 = &l1->items[3];
      const PyObject *b3 = &l2->items[3];
      if (a3->type_id == list_type_id || b3->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a3, b3, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 4)
    {
      const PyObject *a4 = &l1->items[4];
      const PyObject *b4 = &l2->items[4];
      if (a4->type_id == list_type_id || b4->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a4, b4, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 5)
    {
      const PyObject *a5 = &l1->items[5];
      const PyObject *b5 = &l2->items[5];
      if (a5->type_id == list_type_id || b5->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a5, b5, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 6)
    {
      const PyObject *a6 = &l1->items[6];
      const PyObject *b6 = &l2->items[6];
      if (a6->type_id == list_type_id || b6->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a6, b6, list_type_id, string_type_id))
        return false;
    }
    if (l1->size > 7)
    {
      const PyObject *a7 = &l1->items[7];
      const PyObject *b7 = &l2->items[7];
      if (a7->type_id == list_type_id || b7->type_id == list_type_id)
        return false;
      if (!__ESBMC_list_elem_eq_simple(a7, b7, list_type_id, string_type_id))
        return false;
    }
    return true;
  }
  return false;
}

PyObject *__ESBMC_list_at(PyListObject *l, size_t index)
{
  __ESBMC_assert(index < l->size, "out-of-bounds read in list");
  return &l->items[index];
}

bool __ESBMC_list_set_at(
  PyListObject *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size)
{
  __ESBMC_assert(l != NULL, "list_set_at: list is null");
  __ESBMC_assert(index < l->size, "list_set_at: index out of bounds");

  // Make a copy of the new value
  void *copied_value = __ESBMC_alloca(type_size);
  memcpy(copied_value, value, type_size);

  // Update the element at the given index
  l->items[index].value = copied_value;
  l->items[index].type_id = type_id;
  l->items[index].size = type_size;

  return true;
}

bool __ESBMC_list_insert(
  PyListObject *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size)
{
  // If index is beyond the end, just append
  if (index >= l->size)
    return __ESBMC_list_push(l, value, type_id, type_size);

  // Make a copy of the value
  void *copied_value = __ESBMC_alloca(type_size);
  memcpy(copied_value, value, type_size);

  // TODO: there oughta be a better way to do this
  size_t i = l->size;
  while (i > index)
  {
    l->items[i] = l->items[i - 1];
    i--;
  }

  // Insert the new element
  l->items[index].value = copied_value;
  l->items[index].type_id = type_id;
  l->items[index].size = type_size;
  l->size++;
  return true;
}

bool __ESBMC_list_contains(
  const PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size,
  size_t string_type_id)
{
  if (!l || !item)
    return false;

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    // String comparison: compare by content, ignore size mismatches
    if (elem->type_id == string_type_id && item_type_id == string_type_id)
    {
      const char *elem_str = __ESBMC_list_elem_to_str(elem);
      const char *item_str = __ESBMC_list_item_to_str(item, item_size);
      if (elem_str && item_str &&
          __ESBMC_str_equal_bounded(elem_str, item_str, 64))
        return true;
    }
    else if (elem->type_id == item_type_id && elem->size == item_size)
    {
      if (__ESBMC_values_equal(elem->value, item, item_size))
        return true;
    }

    ++i;
  }
  return false;
}

/* ---------- extend list ---------- */

void __ESBMC_list_extend(PyListObject *l, const PyListObject *other)
{
  if (!l || !other)
    return;

  size_t i = 0;
  while (i < other->size)
  {
    const PyObject *elem = &other->items[i];

    void *copied_value = __ESBMC_alloca(elem->size);
    memcpy(copied_value, elem->value, elem->size);

    l->items[l->size].value = copied_value;
    l->items[l->size].type_id = elem->type_id;
    l->items[l->size].size = elem->size;
    l->size++;

    ++i;
  }
}

void __ESBMC_list_clear(PyListObject *l)
{
  if (!l)
    return;
  l->size = 0;
}

size_t __ESBMC_list_find_index(
  PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size,
  size_t string_type_id)
{
  __ESBMC_assert(l != NULL, "KeyError: dictionary is null");
  __ESBMC_assert(item != NULL, "KeyError: key is null");
  __ESBMC_assert(l->size > 0, "KeyError: dictionary is empty");

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    if (elem->type_id == string_type_id && item_type_id == string_type_id)
    {
      const char *elem_str = __ESBMC_list_elem_to_str(elem);
      const char *item_str = __ESBMC_list_item_to_str(item, item_size);
      if (elem_str && item_str &&
          __ESBMC_str_equal_bounded(elem_str, item_str, 64))
        return i;
    }
    else if (elem->type_id == item_type_id && elem->size == item_size)
    {
      if (__ESBMC_values_equal(elem->value, item, item_size))
        return i;
    }

    i = i + 1;
  }

  __ESBMC_assert(0, "KeyError: key not found in dictionary");
  return 0;
}

size_t __ESBMC_list_try_find_index(
  PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size,
  size_t string_type_id)
{
  if (!l || !item || l->size == 0)
    return SIZE_MAX;

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    if (elem->type_id == string_type_id && item_type_id == string_type_id)
    {
      const char *elem_str = __ESBMC_list_elem_to_str(elem);
      const char *item_str = __ESBMC_list_item_to_str(item, item_size);
      if (elem_str && item_str &&
          __ESBMC_str_equal_bounded(elem_str, item_str, 64))
        return i;
    }
    else if (elem->type_id == item_type_id && elem->size == item_size)
    {
      if (__ESBMC_values_equal(elem->value, item, item_size))
        return i;
    }

    i = i + 1;
  }

  return SIZE_MAX; // Not found
}

bool __ESBMC_list_remove_at(PyListObject *l, size_t index)
{
  __ESBMC_assert(l != NULL, "list_remove_at: list is null");
  __ESBMC_assert(index < l->size, "list_remove_at: index out of bounds");

  // Shift elements to fill the gap
  size_t i = index;
  while (i < l->size - 1)
  {
    l->items[i] = l->items[i + 1];
    i++;
  }

  // Decrease size
  l->size = l->size - 1;
  return true;
}

PyObject *__ESBMC_list_pop(PyListObject *l, int64_t index)
{
  __ESBMC_assert(l != NULL, "IndexError: pop from empty list");
  __ESBMC_assert(l->size > 0, "IndexError: pop from empty list");

  // Handle negative index or default (pop last element)
  size_t actual_index;
  if (index < 0)
  {
    // Convert negative index to positive
    int64_t positive_index = (int64_t)l->size + index;
    __ESBMC_assert(positive_index >= 0, "IndexError: pop index out of range");
    actual_index = (size_t)positive_index;
  }
  else
    actual_index = (size_t)index;

  __ESBMC_assert(actual_index < l->size, "IndexError: pop index out of range");

  // Make a copy of the element to return before shifting
  PyObject *popped = __ESBMC_alloca(sizeof(PyObject));

  // Copy the element's data
  popped->value = __ESBMC_alloca(l->items[actual_index].size);
  memcpy(
    (void *)popped->value,
    l->items[actual_index].value,
    l->items[actual_index].size);
  popped->type_id = l->items[actual_index].type_id;
  popped->size = l->items[actual_index].size;

  // Now shift elements to fill the gap
  size_t i = actual_index;
  while (i < l->size - 1)
  {
    l->items[i] = l->items[i + 1];
    i++;
  }

  // Decrease size
  l->size = l->size - 1;

  return popped;
}

bool __ESBMC_dict_eq(
  const PyListObject *lhs_keys,
  const PyListObject *lhs_values,
  const PyListObject *rhs_keys,
  const PyListObject *rhs_values,
  size_t string_type_id)
{
  if (!lhs_keys || !lhs_values || !rhs_keys || !rhs_values)
    return false;

  // Sizes must match
  if (lhs_keys->size != rhs_keys->size)
    return false;

  // For each key-value pair in lhs, check if it exists in rhs
  size_t i = 0;
  while (i < lhs_keys->size)
  {
    const PyObject *lhs_key = &lhs_keys->items[i];
    const PyObject *lhs_value = &lhs_values->items[i];

    // Find this key in rhs_keys
    size_t rhs_idx = __ESBMC_list_try_find_index(
      (PyListObject *)rhs_keys,
      lhs_key->value,
      lhs_key->type_id,
      lhs_key->size,
      string_type_id);

    // Key not found in rhs
    if (rhs_idx == SIZE_MAX)
      return false;

    // Key found: compare the corresponding values
    const PyObject *rhs_value = &rhs_values->items[rhs_idx];

    // Values must have same type and size
    if (
      lhs_value->type_id != rhs_value->type_id ||
      lhs_value->size != rhs_value->size)
      return false;

    // Compare actual value contents
    if (!__ESBMC_values_equal(
          lhs_value->value, rhs_value->value, lhs_value->size))
      return false;

    i++;
  }

  return true;
}
