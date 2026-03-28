#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include "list.h"

// TODO: There is no such a thing as a generic type in python.
static PyType __ESBMC_generic_type;
static PyType __ESBMC_list_type;

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

// Store a dict pointer directly in the list without byte-copying.
// Used for nested dicts so that pointer identity is preserved in the SMT model.
bool __ESBMC_list_push_dict_ptr(PyListObject *l, void *dict_ptr, size_t type_id)
{
  PyObject *item = &l->items[l->size];
  item->value = dict_ptr;
  item->type_id = type_id;
  item->size = 0;
  l->size++;
  return true;
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
      if (a->size == 0 || !__ESBMC_values_equal(a->value, b->value, a->size))
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

    // Copy the value
    void *copied_value = __ESBMC_copy_value(elem->value, elem->size);

    // Add to new list
    copied->items[copied->size].value = copied_value;
    copied->items[copied->size].type_id = elem->type_id;
    copied->items[copied->size].size = elem->size;
    copied->size++;

    ++i;
  }

  return copied;
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
  __ESBMC_assert(0, "ValueError: list.remove(x): x not in list");
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
