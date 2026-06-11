#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>
#include "python_types.h"

// TODO: There is no such a thing as a generic type in python.
static PyType __ESBMC_generic_type;
static PyType __ESBMC_list_type;

// In --ir (integer/real arithmetic) mode, __ESBMC_alloca returns byte-sorted
// storage; storing a float there truncates the real value. To avoid this, we
// copy float values into this global double array (real-sorted, never expires).
#define __ESBMC_FLOAT_BUF_SIZE 4096
static double __ESBMC_float_buf[__ESBMC_FLOAT_BUF_SIZE];
static size_t __ESBMC_float_buf_idx = 0;

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
  if (size == 16)
    return ((const uint64_t *)a)[0] == ((const uint64_t *)b)[0] &&
           ((const uint64_t *)a)[1] == ((const uint64_t *)b)[1];
  // Fallback for larger/unusual sizes. A word-wise compare loop here would
  // unwind --unwind times on every symbolic-size comparison, with no benefit
  // to any converging test (large-struct compares only occur in tests that
  // stay KNOWNBUG on the symbolic-list scalability wall, #5121).
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

// ptr_free=1: payload has no pointer field, so we can reinterpret it as
// uint64_t and use the 16/24/32-byte fast paths below. Pass 0 for opaque or
// pointer-bearing structs — writing through a uint64_t lvalue would lose the
// pointer object under ESBMC's byte-encoding.
static inline void *__ESBMC_copy_value(
  const void *value,
  size_t size,
  size_t type_id,
  size_t float_type_id,
  size_t *out_float_idx,
  int ptr_free)
{
  if (out_float_idx)
    *out_float_idx = 0;

  // None type (NULL pointer with size 0)
  // Don't allocate: return NULL to preserve None semantics
  if (value == NULL && size == 0)
    return NULL;

  // In --ir mode, __ESBMC_alloca returns byte-sorted storage; writing a float
  // (real sort) through any integer cast truncates it. Copy into the global
  // double buffer instead: it is real-sorted and never expires, so the stored
  // pointer stays valid regardless of the caller's scope.
  if (size == 8 && float_type_id != 0 && type_id == float_type_id)
  {
    size_t idx = __ESBMC_float_buf_idx++;
    __ESBMC_float_buf[idx] = *(const double *)value;
    if (out_float_idx)
      *out_float_idx = idx;
    return (void *)&__ESBMC_float_buf[idx];
  }

  void *copied = __ESBMC_alloca(size);

  // Branch-free 8-byte-aligned fast paths for the common small sizes. These
  // avoid memcpy's per-byte loop, which blows up incremental-bmc (size unwind
  // iterations per copied element) and, under a tight --unwind, trips the copy
  // loop's unwinding assertion (dict_tuple_key copies a 3-int tuple key at
  // --unwind 3, #4805). Larger payloads fall through to memcpy: a word-wise
  // loop here would unwind --unwind times on every call where size is symbolic
  // (e.g. the list_slice_assign snapshot loop), on top of memcpy's own loop,
  // pushing list-slice-assign past the CI per-test cap for no benefit to any
  // converging test (large-struct copies only appear in tests that stay
  // KNOWNBUG on the symbolic-list scalability wall, #5121).
  if (size == 8)
    *(uint64_t *)copied = *(const uint64_t *)value;
  else if (size == 16)
  {
    ((uint64_t *)copied)[0] = ((const uint64_t *)value)[0];
    ((uint64_t *)copied)[1] = ((const uint64_t *)value)[1];
  }
  else if (ptr_free && size == 24)
  {
    ((uint64_t *)copied)[0] = ((const uint64_t *)value)[0];
    ((uint64_t *)copied)[1] = ((const uint64_t *)value)[1];
    ((uint64_t *)copied)[2] = ((const uint64_t *)value)[2];
  }
  else if (ptr_free && size == 32)
  {
    ((uint64_t *)copied)[0] = ((const uint64_t *)value)[0];
    ((uint64_t *)copied)[1] = ((const uint64_t *)value)[1];
    ((uint64_t *)copied)[2] = ((const uint64_t *)value)[2];
    ((uint64_t *)copied)[3] = ((const uint64_t *)value)[3];
  }
  else
    memcpy(copied, value, size);

  return copied;
}

bool __ESBMC_list_push(
  PyListObject *l,
  const void *value,
  size_t type_id,
  size_t type_size,
  size_t float_type_id,
  int ptr_free)
{
  // TODO: __ESBMC_obj_cpy
  size_t float_idx = 0;
  void *copied_value = __ESBMC_copy_value(
    value, type_size, type_id, float_type_id, &float_idx, ptr_free);

  // Use a pointer to avoid repeated indexing
  PyObject *item = &l->items[l->size];
  item->value = copied_value;
  item->float_idx = float_idx;
  item->type_id = type_id;
  item->size = type_size;
  l->size++;

  // TODO: Nondeterministic failure?
  return true;
}

bool __ESBMC_list_push_object(
  PyListObject *l,
  PyObject *o,
  size_t float_type_id,
  int ptr_free)
{
  assert(l != NULL);
  assert(o != NULL);
  // For float elements, read from the global float_buf array via a local temp.
  // This avoids the expired-pointer issue of loop-scoped $list_elem symbols,
  // and the local temp ensures the pointer is "fresh" (not stored void*) in --ir.
  if (o->size == 8 && float_type_id != 0 && o->type_id == float_type_id)
  {
    double temp = __ESBMC_float_buf[o->float_idx];
    return __ESBMC_list_push(
      l, (const void *)&temp, o->type_id, o->size, float_type_id, ptr_free);
  }
  return __ESBMC_list_push(
    l, o->value, o->type_id, o->size, float_type_id, ptr_free);
}

// Per-element append for list copy / assignment / slice / concat that handles
// elements stored by reference correctly (esbmc/esbmc#5102).
//
// An element whose payload is a pointer to a shared object must keep that
// pointer, not have its pointee byte-copied. Two such elements exist:
//   * nested lists — the inner PyListObject* is stored in `value`
//     (type_id == list_type_id);
//   * pointer-only payloads stored with size == 0 — e.g. nested dicts inserted
//     via __ESBMC_list_push_dict_ptr (value holds the dict*), and None
//     (value == NULL).
// For these we copy the PyObject record verbatim, preserving the pointer. The
// generic byte-copy would run __ESBMC_copy_value with size == 0 (alloca(0) +
// memcpy of 0 bytes) and drop the stored pointer, so the copied list would read
// garbage. Sharing the reference also matches Python's shallow-copy semantics:
// nested containers are shared, not deep-copied.
//
// Scalar elements (size > 0) must NOT share their buffer: Python subscript
// assignment (l[i] = x) writes through the element's value pointer in place, so
// two lists sharing a scalar buffer would alias. Scalars therefore keep the
// independent byte-copy via __ESBMC_list_push_object.
bool __ESBMC_list_push_shallow(
  PyListObject *l,
  PyObject *o,
  size_t list_type_id)
{
  assert(l != NULL);
  assert(o != NULL);
  if (o->size == 0 || (list_type_id != 0 && o->type_id == list_type_id))
  {
    l->items[l->size] = *o;
    l->size++;
    return true;
  }
  return __ESBMC_list_push_object(l, o, 0, 0);
}

// Store a dict pointer directly in the list without byte-copying.
// Used for nested dicts so that pointer identity is preserved in the SMT model.
bool __ESBMC_list_push_dict_ptr(PyListObject *l, void *dict_ptr, size_t type_id)
{
  PyObject *item = &l->items[l->size];
  item->value = dict_ptr;
  item->float_idx = 0;
  item->type_id = type_id;
  item->size = 0;
  l->size++;
  return true;
}

bool __ESBMC_list_eq(
  const PyListObject *l1,
  const PyListObject *l2,
  size_t list_type_id,
  size_t max_depth,
  size_t float_type_id)
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
    if (a->size != b->size)
      return false;

    // When type IDs differ (e.g., void*/Any vs int from recursive generator
    // parameter type erasure), fall back to byte-wise value comparison.
    // This is safe because Any elements store the same bit pattern as the
    // original concrete-typed elements (the value, not a pointer to it).
    if (a->type_id != b->type_id)
    {
      // size == 0 means a dict pointer or None element — stored by pointer
      // identity, not byte content, so cross-type-id comparison is unsound.
      if (a->size == 0)
        return false;

      // int-vs-float: Python compares numerically (1 == 1.0), but the two
      // store different bit patterns, so a byte compare would wrongly differ.
      // Triggered when exactly one side is a float (sizes already match here,
      // and float/int elements are both 8 bytes). float_type_id is derived by
      // the frontend from top-level element types only, so this numeric path
      // does not reach nested mixed int/float lists; like list_lt, the int side
      // is widened to double, so |int| > 2^53 loses precision vs CPython.
      if (
        float_type_id != 0 && a->size == 8 &&
        (a->type_id == float_type_id) != (b->type_id == float_type_id))
      {
        double av = (a->type_id == float_type_id)
                      ? *(const double *)a->value
                      : (double)*(const int64_t *)a->value;
        double bv = (b->type_id == float_type_id)
                      ? *(const double *)b->value
                      : (double)*(const int64_t *)b->value;
        if (av != bv)
          return false;
        continue;
      }

      if (!__ESBMC_values_equal(a->value, b->value, a->size))
        return false;
      continue;
    }

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
  size_t type_size,
  size_t float_type_id,
  int ptr_free)
{
  __ESBMC_assert(l != NULL, "list_set_at: list is null");
  __ESBMC_assert(index < l->size, "list_set_at: index out of bounds");

  size_t float_idx = 0;
  void *copied_value = __ESBMC_copy_value(
    value, type_size, type_id, float_type_id, &float_idx, ptr_free);

  // Update the element at the given index
  l->items[index].value = copied_value;
  l->items[index].float_idx = float_idx;
  l->items[index].type_id = type_id;
  l->items[index].size = type_size;

  return true;
}

bool __ESBMC_list_insert(
  PyListObject *l,
  size_t index,
  const void *value,
  size_t type_id,
  size_t type_size,
  size_t float_type_id,
  int ptr_free)
{
  // If index is beyond the end, just append
  if (index >= l->size)
    return __ESBMC_list_push(
      l, value, type_id, type_size, float_type_id, ptr_free);

  size_t float_idx = 0;
  void *copied_value = __ESBMC_copy_value(
    value, type_size, type_id, float_type_id, &float_idx, ptr_free);

  // TODO: there oughta be a better way to do this
  size_t i = l->size;
  while (i > index)
  {
    l->items[i] = l->items[i - 1];
    i--;
  }

  // Insert the new element
  l->items[index].value = copied_value;
  l->items[index].float_idx = float_idx;
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

size_t __ESBMC_list_count(
  const PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  if (!l || !item)
    return 0;

  size_t cnt = 0;
  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];
    if (
      elem->type_id == item_type_id && elem->size == item_size &&
      __ESBMC_values_equal(elem->value, item, item_size))
      ++cnt;
    ++i;
  }
  return cnt;
}

size_t __ESBMC_list_index(
  const PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  if (!l || !item)
    return 0;

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];
    if (
      elem->type_id == item_type_id && elem->size == item_size &&
      __ESBMC_values_equal(elem->value, item, item_size))
      return i;
    ++i;
  }
  __ESBMC_assert(0, "ValueError: list.index(x): x not in list");
  return 0;
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

    // Reuse the float-aware copier so the SMT model tracks size.
    void *copied_value =
      __ESBMC_copy_value(elem->value, elem->size, elem->type_id, 0, NULL, 0);

    l->items[l->size].value = copied_value;
    l->items[l->size].float_idx = elem->float_idx;
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

// Fast-equal for dict-key lookup. Adds 8-byte-aligned fast paths for tuple
// keys (sizes 16/24/32) on top of __ESBMC_values_equal's scalar paths.
// Kept local to list_try_find_index so that generic values_equal callers
// (list_eq, list_contains, etc.) are not penalised by extra branches.
static inline bool __ESBMC_key_equal(const void *a, const void *b, size_t size)
{
  if (a == b)
    return true;
  if (size == 8)
    return *(const uint64_t *)a == *(const uint64_t *)b;
  if (size == 1)
    return *(const uint8_t *)a == *(const uint8_t *)b;
  if (size == 16)
    return ((const uint64_t *)a)[0] == ((const uint64_t *)b)[0] &&
           ((const uint64_t *)a)[1] == ((const uint64_t *)b)[1];
  if (size == 24)
    return ((const uint64_t *)a)[0] == ((const uint64_t *)b)[0] &&
           ((const uint64_t *)a)[1] == ((const uint64_t *)b)[1] &&
           ((const uint64_t *)a)[2] == ((const uint64_t *)b)[2];
  if (size == 32)
    return ((const uint64_t *)a)[0] == ((const uint64_t *)b)[0] &&
           ((const uint64_t *)a)[1] == ((const uint64_t *)b)[1] &&
           ((const uint64_t *)a)[2] == ((const uint64_t *)b)[2] &&
           ((const uint64_t *)a)[3] == ((const uint64_t *)b)[3];
  return memcmp(a, b, size) == 0;
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
      if (__ESBMC_key_equal(elem->value, item, item_size))
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

  // Return the removed object as-is. The payload already has stable storage
  // and shifting the remaining slots only moves PyObject descriptors.
  *popped = l->items[actual_index];

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

    // None values are stored as (value=NULL, size=0).
    // Treat them separately from the size==0 list-pointer path below,
    // which would otherwise dereference NULL.
    const bool lhs_is_none = (lhs_value->value == NULL && lhs_value->size == 0);
    const bool rhs_is_none = (rhs_value->value == NULL && rhs_value->size == 0);
    if (lhs_is_none || rhs_is_none)
    {
      if (lhs_is_none != rhs_is_none)
        return false;
    }
    // setdefault list (size==0, raw PyListObject*) vs literal list (size>0,
    // pointer-to-PyListObject*): normalise both sides and use list_eq.
    // Restricted to asymmetric size to leave symmetric size==0 payloads
    // (e.g. nested dict pointers) on the byte-compare path.
    else if ((lhs_value->size == 0) != (rhs_value->size == 0))
    {
      const PyListObject *lhs_list =
        (lhs_value->size == 0) ? (const PyListObject *)lhs_value->value
                               : *(const PyListObject **)lhs_value->value;
      const PyListObject *rhs_list =
        (rhs_value->size == 0) ? (const PyListObject *)rhs_value->value
                               : *(const PyListObject **)rhs_value->value;
      if (!__ESBMC_list_eq(lhs_list, rhs_list, 0, 0, 0))
        return false;
    }
    else
    {
      // type_id alone does not pin size for variable-sized payloads
      // (e.g. strings: same type_id, size == strlen+1).
      if (
        lhs_value->type_id != rhs_value->type_id ||
        lhs_value->size != rhs_value->size)
        return false;
      if (!__ESBMC_values_equal(
            lhs_value->value, rhs_value->value, lhs_value->size))
        return false;
    }

    i++;
  }

  return true;
}

PyListObject *__ESBMC_list_copy(const PyListObject *l)
{
  if (!l)
    return NULL;

  // Create new list
  PyListObject *copied = __ESBMC_list_create();

  // Copy all elements
  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    // Generic per-element copy: float_type_id=0 disables float-aware copy,
    // ptr_free=0 routes through memcpy (we don't track per-element types).
    void *copied_value =
      __ESBMC_copy_value(elem->value, elem->size, 0, 0, NULL, 0);

    // Add to new list
    copied->items[copied->size].value = copied_value;
    copied->items[copied->size].float_idx = elem->float_idx;
    copied->items[copied->size].type_id = elem->type_id;
    copied->items[copied->size].size = elem->size;
    copied->size++;

    ++i;
  }

  return copied;
}

// Store `o` into an existing slot, with __ESBMC_list_push_shallow's sharing
// rules: pointer-payload elements (nested lists/dicts/None, size == 0 or
// type_id == list_type_id) keep their pointer record; scalars get an
// independent byte-copy so two slots never alias one buffer.
static void
__ESBMC_list_store_elem(PyObject *slot, const PyObject *o, size_t list_type_id)
{
  if (o->size == 0 || (list_type_id != 0 && o->type_id == list_type_id))
  {
    *slot = *o;
    return;
  }
  slot->value = __ESBMC_copy_value(o->value, o->size, o->type_id, 0, NULL, 0);
  slot->float_idx = o->float_idx;
  slot->type_id = o->type_id;
  slot->size = o->size;
}

// CPython slice assignment: l[lower:upper:step] = src.
// has_lower/has_upper distinguish an absent bound from an explicit one;
// present bounds follow slice.indices(len(l)) normalization (negative bounds
// add len, then clamp). step == 1 may resize the list (replace [start, stop)
// with all of src); any other step requires len(src) == slice length, as in
// CPython, which raises ValueError otherwise (modelled as a failing assert,
// like the step-zero case).
bool __ESBMC_list_slice_assign(
  PyListObject *l,
  int64_t lower,
  int has_lower,
  int64_t upper,
  int has_upper,
  int64_t step,
  const PyListObject *src,
  size_t list_type_id)
{
  __ESBMC_assert(
    l != NULL && src != NULL, "list_slice_assign: list or source is null");
  __ESBMC_assert(step != 0, "ValueError: slice step cannot be zero");

  int64_t size = (int64_t)l->size;
  int64_t lo_clamp = (step < 0) ? -1 : 0;
  int64_t hi_clamp = (step < 0) ? size - 1 : size;

  int64_t start;
  if (has_lower)
  {
    start = (lower < 0) ? lower + size : lower;
    if (start < lo_clamp)
      start = lo_clamp;
    if (start > hi_clamp)
      start = hi_clamp;
  }
  else
    start = (step < 0) ? size - 1 : 0;

  int64_t stop;
  if (has_upper)
  {
    stop = (upper < 0) ? upper + size : upper;
    if (stop < lo_clamp)
      stop = lo_clamp;
    if (stop > hi_clamp)
      stop = hi_clamp;
  }
  else
    stop = (step < 0) ? -1 : size;

  int64_t slicelen;
  if (step < 0)
    slicelen = (stop < start) ? (start - stop - 1) / (-step) + 1 : 0;
  else
    slicelen = (start < stop) ? (stop - start - 1) / step + 1 : 0;

  int64_t srclen = (int64_t)src->size;

  // Self-assignment (l[1:] = l): snapshot src before mutating l. The snapshot
  // must follow the same sharing rules as the writes below: a generic
  // __ESBMC_list_copy would byte-copy pointer-payload elements (nested
  // lists/dicts/None, size == 0) and drop the stored pointer (#5102).
  if (src == l)
  {
    PyListObject *snap = __ESBMC_list_create();
    int64_t i = 0;
    while (i < size)
    {
      __ESBMC_list_push_shallow(snap, &l->items[i], list_type_id);
      i++;
    }
    src = snap;
  }

  if (step == 1)
  {
    if (srclen < slicelen)
    {
      // Shrink: shift the tail left to close the gap.
      int64_t to = start + srclen;
      int64_t from = start + slicelen;
      while (from < size)
        l->items[to++] = l->items[from++];
    }
    else if (srclen > slicelen)
    {
      // Grow: shift the tail right, last element first.
      int64_t shift = srclen - slicelen;
      int64_t i = size - 1;
      while (i >= start + slicelen)
      {
        l->items[i + shift] = l->items[i];
        i--;
      }
    }
    l->size = (size_t)(size + srclen - slicelen);
  }
  else
    __ESBMC_assert(
      srclen == slicelen,
      "ValueError: attempt to assign sequence of different size to extended "
      "slice");

  // For step != 1 a length mismatch has already failed the assert above;
  // still bound the writes by the slice so no slot outside it is touched.
  int64_t writelen = srclen;
  if (step != 1 && slicelen < writelen)
    writelen = slicelen;

  int64_t k = 0;
  int64_t idx = start;
  while (k < writelen)
  {
    __ESBMC_list_store_elem(&l->items[idx], &src->items[k], list_type_id);
    idx += step;
    k++;
  }

  return true;
}

bool __ESBMC_list_remove(
  PyListObject *l,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  __ESBMC_assert(l != NULL, "ValueError: list is null");

  size_t i = 0;
  while (i < l->size)
  {
    const PyObject *elem = &l->items[i];

    if (elem->type_id == item_type_id && elem->size == item_size)
    {
      if (__ESBMC_values_equal(elem->value, item, item_size))
      {
        /* Shift elements left to fill the gap */
        size_t j = i;
        while (j < l->size - 1)
        {
          l->items[j] = l->items[j + 1];
          j++;
        }
        l->size--;
        return true; /* found and removed */
      }
    }
    i++;
  }

  /* Item not found */
  return false;
}

/* set.add(elem) — append elem to the underlying list iff it is not
 * already present. Returns true when the set was modified. */
bool __ESBMC_set_add(
  PyListObject *s,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  __ESBMC_assert(s != NULL, "ValueError: set is null");

  if (__ESBMC_list_contains(s, item, item_type_id, item_size))
    return false;

  return __ESBMC_list_push(s, item, item_type_id, item_size, 0, 0);
}

/* set.discard(elem) — like list.remove but silent when the element is
 * absent. Returns true when an element was removed. */
bool __ESBMC_set_discard(
  PyListObject *s,
  const void *item,
  size_t item_type_id,
  size_t item_size)
{
  __ESBMC_assert(s != NULL, "ValueError: set is null");

  size_t i = 0;
  while (i < s->size)
  {
    const PyObject *elem = &s->items[i];

    if (elem->type_id == item_type_id && elem->size == item_size)
    {
      if (__ESBMC_values_equal(elem->value, item, item_size))
      {
        size_t j = i;
        while (j < s->size - 1)
        {
          s->items[j] = s->items[j + 1];
          j++;
        }
        s->size--;
        return true;
      }
    }
    i++;
  }

  return false;
}

void __ESBMC_list_sort(PyListObject *l, int type_flag, uint64_t float_type_id)
{
  if (!l || l->size <= 1)
    return;

  size_t n = l->size;

  size_t i = 1;
  while (i < n)
  {
    PyObject tmp = l->items[i];
    size_t j = i;

    while (j > 0)
    {
      PyObject *prev = &l->items[j - 1];

      // For numeric types both sizes must match (same storage width).
      // For strings sizes may differ ("apple"=6 vs "banana"=7), so only
      // reject mismatched sizes for non-string (non-type_flag-2) lists.
      if (prev->size != tmp.size && type_flag != 2)
        break;

      bool prev_greater = false;

      if (prev->size == 8 && type_flag == 0)
      {
        // All-integer list: compare as int64_t.
        // Stays entirely in integer arithmetic — fast for the SMT solver.
        int64_t a = *(const int64_t *)prev->value;
        int64_t b = *(const int64_t *)tmp.value;
        prev_greater = (a > b);
      }
      else if (prev->size == 8 && type_flag == 1)
      {
        // All-float list: read bits directly as IEEE 754 double.
        double a = *(const double *)prev->value;
        double b = *(const double *)tmp.value;
        prev_greater = (a > b);
      }
      else if (prev->size == 8 && type_flag == 3)
      {
        // Mixed int + float list.
        // Per-element dispatch: check each element's own type_id.
        //   float element → read bits as double
        //   int element   → numeric cast (double)(int64_t)  [exact up to 2^53]
        double a = (prev->type_id == float_type_id)
                     ? (*(const double *)prev->value)
                     : ((double)(*(const int64_t *)prev->value));
        double b = (tmp.type_id == float_type_id)
                     ? (*(const double *)tmp.value)
                     : ((double)(*(const int64_t *)tmp.value));
        prev_greater = (a > b);
      }
      else if (prev->size == 1)
      {
        // bool / single-byte
        uint8_t a = *(const uint8_t *)prev->value;
        uint8_t b = *(const uint8_t *)tmp.value;
        prev_greater = (a > b);
      }
      else
      {
        // type_flag == 2: string / lexicographic comparison.
        //
        // Must use min(prev->size, tmp->size) as the memcmp length.
        // Using prev->size alone reads past the end of tmp's buffer when
        // prev is longer than tmp — ESBMC models that out-of-bounds byte as
        // a symbolic value, making the comparison nondeterministic.
        //
        // After the shared prefix compares equal, the shorter string is
        // lesser (matching Python / C string ordering).
        size_t min_size = (prev->size < tmp.size) ? prev->size : tmp.size;
        int cmp = memcmp(prev->value, tmp.value, min_size);
        if (cmp == 0)
          cmp = (prev->size > tmp.size) - (prev->size < tmp.size);
        prev_greater = (cmp > 0);
      }

      if (!prev_greater)
        break;

      l->items[j] = l->items[j - 1];
      j--;
    }

    l->items[j] = tmp;
    i++;
  }
}

// Lexicographic less-than for Python lists.
// type_flag:  0=int  1=float  2=str  3=mixed-int+float  (same encoding as
//             __ESBMC_list_sort)
// float_type_id: type_id hash for float elements (used when type_flag == 3)
//
// Returns true iff l1 < l2 under Python lexicographic ordering:
//   1. Compare element by element; the first unequal pair decides.
//   2. If all shared elements are equal, the shorter list is smaller.
bool __ESBMC_list_lt(
  const PyListObject *l1,
  const PyListObject *l2,
  int type_flag,
  size_t float_type_id)
{
  if (!l1 || !l2)
    return false;

  size_t n = l1->size < l2->size ? l1->size : l2->size;
  size_t i = 0;
  while (i < n)
  {
    const PyObject *a = &l1->items[i];
    const PyObject *b = &l2->items[i];
    i++;

    // Same pointer → elements are identical
    if (a->value == b->value)
      continue;

    if (type_flag == 2)
    {
      // String / lexicographic comparison.  a->size / b->size include the
      // null terminator (matching __ESBMC_list_sort convention).  Using
      // min_size (which includes the null byte of the shorter string) in
      // memcmp is safe: when two strings have the same content up to that
      // length, the null byte in the shorter string compares as 0 < any
      // real character, giving the correct "shorter < longer" result.
      size_t min_size = a->size < b->size ? a->size : b->size;
      int cmp = memcmp(a->value, b->value, min_size);
      if (cmp != 0)
        return cmp < 0;
      if (a->size != b->size)
        return a->size < b->size;
      // Strings are equal; continue to next element.
    }
    else if (a->size == 8 && type_flag == 1)
    {
      double av = *(const double *)a->value;
      double bv = *(const double *)b->value;
      if (av != bv)
        return av < bv;
    }
    else if (a->size == 8 && type_flag == 3)
    {
      double av = (a->type_id == float_type_id)
                    ? *(const double *)a->value
                    : (double)(*(const int64_t *)a->value);
      double bv = (b->type_id == float_type_id) ? *(const double *)b->value
                  : (b->size == 1) ? (double)(*(const uint8_t *)b->value)
                                   : (double)(*(const int64_t *)b->value);
      if (av != bv)
        return av < bv;
    }
    else if (a->size == 8)
    {
      // Integer (type_flag == 0)
      int64_t av = *(const int64_t *)a->value;
      int64_t bv = *(const int64_t *)b->value;
      if (av != bv)
        return av < bv;
    }
    else if (a->size == 1)
    {
      uint8_t av = *(const uint8_t *)a->value;
      uint8_t bv = *(const uint8_t *)b->value;
      if (av != bv)
        return av < bv;
    }
    else
    {
      // Fallback: byte-wise comparison (handles unusual sizes)
      size_t min_size = a->size < b->size ? a->size : b->size;
      int cmp = memcmp(a->value, b->value, min_size);
      if (cmp != 0)
        return cmp < 0;
      if (a->size != b->size)
        return a->size < b->size;
    }
  }

  // All shared elements equal: shorter list is less.
  return l1->size < l2->size;
}

void __ESBMC_list_reverse(PyListObject *l)
{
  if (!l || l->size <= 1)
    return;

  size_t left = 0;
  size_t right = l->size - 1;

  while (left < right)
  {
    /* Swap items[left] and items[right] in place */
    PyObject tmp = l->items[left];
    l->items[left] = l->items[right];
    l->items[right] = tmp;

    left++;
    right--;
  }
}
