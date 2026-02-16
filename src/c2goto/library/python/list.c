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
  if (size == 1)
    return *(const uint8_t *)a == *(const uint8_t *)b;
  // Fallback for larger/unusual sizes
  return memcmp(a, b, size) == 0;
}

// Default maximum nesting depth to prevent state explosion during symbolic execution.
// This can be overridden via --python-list-compare-depth option.
#define __ESBMC_LIST_DEFAULT_DEPTH 4

// Maximum physical stack size for list comparison (prevents buffer overflow).
// Set to 64 to allow users to increase depth without hitting buffer limits.
#define __ESBMC_LIST_MAX_STACK 64

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
  size_t max_depth)
{
  // Quick checks
  if (!l1 || !l2)
    return false;
  if (__ESBMC_same_object(l1, l2))
    return true;
  if (l1->size != l2->size)
    return false;

  // Use max_depth or default if 0, but cap at physical stack size
  size_t depth_limit = max_depth > 0 ? max_depth : __ESBMC_LIST_DEFAULT_DEPTH;
  if (depth_limit > __ESBMC_LIST_MAX_STACK)
    depth_limit = __ESBMC_LIST_MAX_STACK;

  // Use explicit stack to avoid recursive function calls
  // This prevents state explosion from recursive unrolling
  const PyListObject *stack_a[__ESBMC_LIST_MAX_STACK];
  const PyListObject *stack_b[__ESBMC_LIST_MAX_STACK];
  size_t stack_idx[__ESBMC_LIST_MAX_STACK];
  int top = 0;

  // Initialize with first list pair
  stack_a[0] = l1;
  stack_b[0] = l2;
  stack_idx[0] = 0;
  top = 1;

  while (top > 0)
  {
    int cur = top - 1;
    const PyListObject *cur_a = stack_a[cur];
    const PyListObject *cur_b = stack_b[cur];
    size_t idx = stack_idx[cur];

    // Finished comparing this list pair?
    if (idx >= cur_a->size)
    {
      top--;
      continue;
    }

    // Advance index for next iteration
    stack_idx[cur] = idx + 1;

    const PyObject *a = &cur_a->items[idx];
    const PyObject *b = &cur_b->items[idx];

    // Same pointer => elements equal
    if (a->value == b->value)
      continue;

    // Validation checks
    if (!a->value || !b->value)
      return false;
    if (a->type_id != b->type_id)
      return false;
    if (a->size != b->size)
      return false;

    // Check if elements are nested lists
    if (a->type_id == list_type_id)
    {
      const PyListObject *nested_a = *(const PyListObject **)a->value;
      const PyListObject *nested_b = *(const PyListObject **)b->value;

      // Quick checks for nested lists
      if (!nested_a || !nested_b)
        return false;
      if (__ESBMC_same_object(nested_a, nested_b))
        continue;
      if (nested_a->size != nested_b->size)
        return false;

      // Check depth limit and report if exceeded
      if ((size_t)top >= depth_limit)
      {
        // List depth unwinding assertion: similar to loop unwinding assertions.
        // If this fires, increase depth with --python-list-compare-depth option.
        __ESBMC_assert(
          0,
          "list comparison depth limit exceeded "
          "(use --python-list-compare-depth to increase)");
        // Note: return is needed to stop symbolic execution on this path
        return false;
      }

      // Push nested comparison onto stack
      stack_a[top] = nested_a;
      stack_b[top] = nested_b;
      stack_idx[top] = 0;
      top++;
    }
    else
    {
      // Primitive comparison - use optimized version (no memcmp loop)
      if (!__ESBMC_values_equal(a->value, b->value, a->size))
        return false;
    }
  }
  return true;
}

// Order-insensitive set equality: compare by value only.
bool __ESBMC_list_set_eq(const PyListObject *l1, const PyListObject *l2)
{
  if (!l1 || !l2)
    return false;
  if (__ESBMC_same_object(l1, l2))
    return true;
  if (l1->size != l2->size)
    return false;

  size_t n = l1->size;
  if (n == 0)
    return true;

  // Track which elements in l2 have been matched.
  bool *matched = (bool *)__ESBMC_alloca(n * sizeof(bool));
  size_t i = 0;
  while (i < n)
  {
    matched[i] = false;
    ++i;
  }

  i = 0;
  while (i < n)
  {
    const PyObject *a = &l1->items[i];
    bool found = false;

    size_t j = 0;
    while (j < n)
    {
      if (matched[j])
      {
        ++j;
        continue;
      }

      const PyObject *b = &l2->items[j];
      if (a->size != b->size)
      {
        ++j;
        continue;
      }

      if (__ESBMC_values_equal(a->value, b->value, a->size))
      {
        matched[j] = true;
        found = true;
        break;
      }
      ++j;
    }

    if (!found)
      return false;
    ++i;
  }

  return true;
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
  size_t item_size)
{
  if (!l || !item)
    return false;

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    // Check if types and sizes match
    if (elem->type_id == item_type_id && elem->size == item_size)
    {
      // Compare the actual data
      // TODO: Not sure if this works for recursive types
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
  size_t item_size)
{
  __ESBMC_assert(l != NULL, "KeyError: dictionary is null");
  __ESBMC_assert(item != NULL, "KeyError: key is null");
  __ESBMC_assert(l->size > 0, "KeyError: dictionary is empty");

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    if (elem->type_id == item_type_id && elem->size == item_size)
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
  size_t item_size)
{
  if (!l || !item || l->size == 0)
    return SIZE_MAX;

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    if (elem->type_id == item_type_id && elem->size == item_size)
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
  const PyListObject *rhs_values)
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
      lhs_key->size);

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
