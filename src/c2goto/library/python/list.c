#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>

/** Based on CPython, the idea is to use a PyObject containing type information
 *  while each actual object is explicitly defined.
 */
typedef struct __ESBMC_PyType
{
  const char *tp_name; /* Type name: "module.typename" */
  size_t tp_basicsize; /* Size of instance in bytes */

  // TODO: Extra features (vtables, constructors, members)
} PyType;

// TODO: There is no such a thing as a generic type in python.
static PyType __ESBMC_generic_type;
static PyType __ESBMC_list_type;

/**
 * @brief Minimal representation of a Python-like object.
 *
 * In CPython, PyObject includes only a pointer to its type object. Most
 * user-defined types embed this as their header, allowing any instance to
 * be safely cast to PyObject* for type inspection.
 *
 * @code
 * PyListObject l = {...};
 * for (size_t i = 0; i < l.size; ++i) {
 *   PyObject *o = (PyObject *)l.values[i];
 *   PyType *t = ((PyObject *)o)->type;
 * }
 * @endcode
 *
 * This simplified version keeps both the type information and the data pointer
 * explicit. The long-term goal is to embed only a type pointer, enabling more
 * lightweight polymorphic casts. 
 */
typedef struct __ESBMC_PyObj
{
  const void *value; /**< Pointer to object data */
  size_t type_id;    /**< Hashed or unique type identifier */
  size_t size;       /**< Number of bytes in value */
} PyObject;

/**
 * @brief Minimal representation of a Python-like list object.
 *
 * Example usage in C:
 * @code
 * PyListObject list = {...};
 * for (size_t i = 0; i < list.size; ++i) {
 *   PyObject *item = &list.items[i];
 *   // Access fields like item->type_id or item->value here
 * }
 * @endcode
 */
typedef struct __ESBMC_PyListObj
{
  PyType *type;    /**< &PyListType */
  PyObject *items; /**< Array of PyObject items (SMT infinite array concept) */
  size_t size;     /**< Number of elements currently in use */
} PyListObject;

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
  void *copied = __ESBMC_alloca(size);

  if (size == 8)
    *(uint64_t *)copied = *(const uint64_t *)value;
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

bool __ESBMC_list_eq(const PyListObject *l1, const PyListObject *l2)
{
  if (!l1 || !l2)
    return false;
  if (__ESBMC_same_object(l1, l2))
    return true;
  if (l1->size != l2->size)
    return false;

  size_t i = 0, end = l1->size;

  // BUG: Something weird is happening when I change this while into a FOR
  while (i < end)
  {
    const PyObject *a = &l1->items[i];
    const PyObject *b = &l2->items[i++];

    // Same address => element equal; keep checking the rest.
    if (a->value == b->value)
      continue;

    if (
      !a->value || !b->value || a->type_id != b->type_id ||
      a->size != b->size || memcmp(a->value, b->value, a->size) != 0)
      return false;
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

static inline bool
__ESBMC_values_equal(const void *a, const void *b, size_t size)
{
  if (a == b)
    return true;

  if (size == 8)
    return *(const uint64_t *)a == *(const uint64_t *)b;
  else
    return memcmp(a, b, size) == 0;
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
