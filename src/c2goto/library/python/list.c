#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h> // SIZE_MAX
#include <string.h>

/// Python predicates

typedef void __PyObject;

/**
 * @typedef unaryfunc
 * @brief Unary function pointer type
 * 
 * Function pointer type for unary operations on Python objects.
 * Used for operations that take a single operand.
 * 
 * @param arg The Python object operand
 * @return New reference to the result object, or NULL on error
 * 
 * @note Common uses include:
 *       - Unary arithmetic operators (negation, positive, absolute value)
 *       - Type conversions (int, float, str)
 *       - Iterator protocol (__iter__)
 */
typedef __PyObject *(*unaryfunc)(__PyObject *);

/**
 * @typedef binaryfunc
 * @brief Binary function pointer type
 * 
 * Function pointer type for binary operations on Python objects.
 * Used for operations that take two operands.
 * 
 * @param left The left operand
 * @param right The right operand
 * @return New reference to the result object, or NULL on error
 * 
 * @note Common uses include:
 *       - Binary arithmetic operators (+, -, *, /, //, %, **, etc.)
 *       - Comparison operators
 *       - Bitwise operators (&, |, ^, <<, >>)
 *       - Container operations (subscript, concatenation)
 */
typedef __PyObject *(*binaryfunc)(__PyObject *, __PyObject *);

/**
 * @typedef ternaryfunc
 * @brief Ternary function pointer type
 * 
 * Function pointer type for ternary operations on Python objects.
 * Used for operations that take three operands.
 * 
 * @param arg1 The first operand
 * @param arg2 The second operand
 * @param arg3 The third operand
 * @return New reference to the result object, or NULL on error
 * 
 * @note Primary use: pow(base, exp, mod) for the three-argument power function
 *       Also used for __setitem__ with slice assignment
 */
typedef __PyObject *(*ternaryfunc)(__PyObject *, __PyObject *, __PyObject *);

/**
 * @typedef inquiry
 * @brief Boolean query function pointer type
 * 
 * Function pointer type for predicate/query operations that return
 * a boolean result as an integer.
 * 
 * @param self The Python object to query
 * @return 1 for true, 0 for false, -1 on error
 * 
 * @note Common uses include:
 *       - Boolean conversion (__bool__)
 *       - Container membership testing
 *       - Object state queries
 */
typedef int (*inquiry)(__PyObject *);

/**
 * @typedef lenfunc
 * @brief Length query function pointer type
 * 
 * Function pointer type for obtaining the length/size of a Python object.
 * 
 * @param self The Python object to query
 * @return The length/size of the object, or (size_t)-1 on error
 * 
 * @note Used for implementing __len__() protocol
 *       Typical for containers (lists, tuples, dicts, sets, strings)
 */
typedef size_t (*lenfunc)(__PyObject *);

/**
 * @typedef ssizeargfunc
 * @brief Indexed access function pointer type
 * 
 * Function pointer type for retrieving an item at a specific index.
 * Takes a signed size type for the index.
 * 
 * @param self The container object
 * @param index The index of the item to retrieve (may be negative)
 * @return New reference to the item at the given index, or NULL on error
 * 
 * @note Used for implementing sequence indexing (__getitem__ with integer index)
 *       Negative indices are typically interpreted as offsets from the end
 */
typedef __PyObject *(*ssizeargfunc)(__PyObject *, size_t);

/**
 * @typedef ssizeobjargproc
 * @brief Indexed assignment function pointer type
 * 
 * Function pointer type for setting or deleting an item at a specific index.
 * 
 * @param self The container object
 * @param index The index where the item should be set
 * @param value The value to set, or NULL to delete the item
 * @return 0 on success, -1 on error
 * 
 * @note Used for implementing sequence item assignment and deletion
 *       (__setitem__ and __delitem__ with integer index)
 *       When value is NULL, the item should be deleted
 */
typedef int (*ssizeobjargproc)(__PyObject *, size_t, __PyObject *);

/**
 * @typedef objobjargproc
 * @brief Object-keyed assignment function pointer type
 * 
 * Function pointer type for setting or deleting items using object keys.
 * 
 * @param self The container object (typically a mapping)
 * @param key The key object
 * @param value The value to associate with the key, or NULL to delete
 * @return 0 on success, -1 on error
 * 
 * @note Used for implementing mapping protocols:
 *       - Dictionary item assignment (__setitem__)
 *       - Dictionary item deletion (__delitem__)
 *       When value is NULL, the key-value pair should be deleted
 */
typedef int (*objobjargproc)(__PyObject *, __PyObject *, __PyObject *);

/**
 * @typedef objobjproc
 * @brief Binary object comparison/operation function pointer type
 * 
 * Function pointer type for operations that take two Python object arguments
 * and return an integer result. This is a simpler variant of objobjargproc
 * that does not involve assignment or deletion.
 * 
 * @param arg1 The first Python object
 * @param arg2 The second Python object
 * @return Integer result (semantics depend on the specific operation):
 *         - For comparisons: -1, 0, or 1 (less than, equal, greater than)
 *         - For boolean queries: 1 for true, 0 for false, -1 on error
 *         - Operation-specific integer result on success, -1 on error
 * 
 * @note Common uses include:
 *       - Rich comparison operators (__lt__, __le__, __eq__, __ne__, __gt__, __ge__)
 *       - Binary predicate operations
 *       - Custom comparison implementations
 * 
 * @see objobjargproc for the three-argument variant used in assignment
 * @see binaryfunc for operations returning object pointers
 */
typedef int (*objobjproc)(__PyObject *, __PyObject *);

/**
 * @typedef richcmpfunc
 * @brief Rich comparison function pointer type
 * 
 * Function pointer type for implementing Python's rich comparison methods.
 * 
 * @param self The first Python object operand (usually the calling object)
 * @param other The second Python object operand to compare against
 * @param op An integer specifying the comparison operation, typically one of:
 *        - Py_LT: less than
 *        - Py_LE: less than or equal
 *        - Py_EQ: equal
 *        - Py_NE: not equal
 *        - Py_GT: greater than
 *        - Py_GE: greater than or equal
 * @return New reference to a PyObject representing the result of the comparison,
 *         typically Py_True or Py_False, or NULL on error.
 * 
 * @note This corresponds to the tp_richcompare slot in PyTypeObject.
 *       It allows objects to implement full rich comparison semantics.
 * @see PyObject_RichCompare(), PyTypeObject::tp_richcompare
 */
typedef _Bool (*richcmpfunc)(__PyObject *, __PyObject *, int);

#define Py_LT 0
#define Py_LE 1
#define Py_EQ 2
#define Py_NE 3
#define Py_GT 4
#define Py_GE 5

/* typedef void (*freefunc)(void *); */
/* typedef void (*destructor)(__PyObject *); */
/* typedef __PyObject *(*reprfunc)(__PyObject *); */
/* typedef __PyObject *(*getattrfunc)(__PyObject *, char *); */
/* typedef __PyObject *(*getattrofunc)(__PyObject *, __PyObject *); */
/* typedef int (*setattrfunc)(__PyObject *, char *, __PyObject *); */
/* typedef int (*setattrofunc)(__PyObject *, __PyObject *, __PyObject *); */
/*  */
/* typedef PyObject *(*getiterfunc) (PyObject *); */
/* typedef PyObject *(*iternextfunc) (PyObject *); */
/* typedef PyObject *(*descrgetfunc) (PyObject *, PyObject *, PyObject *); */
/* typedef int (*descrsetfunc) (PyObject *, PyObject *, PyObject *); */
/* typedef int (*initproc)(PyObject *, PyObject *, PyObject *); */
/* typedef PyObject *(*newfunc)(PyTypeObject *, PyObject *, PyObject *); */
/* typedef PyObject *(*allocfunc)(PyTypeObject *, Py_ssize_t); */
//typedef Py_hash_t (*hashfunc)(PyObject *);
//typedef PyObject *(*ssizessizeargfunc)(PyObject *, Py_ssize_t, Py_ssize_t);
//typedef int(*ssizessizeobjargproc)(PyObject *, Py_ssize_t, Py_ssize_t, PyObject *);
//typedef int (*objobjproc)(PyObject *, PyObject *);
//typedef int (*visitproc)(PyObject *, void *);
//typedef int (*traverseproc)(PyObject *, visitproc, void *);

// PySequence

/**
 * @struct PySequenceMethods
 * @brief Sequence protocol implementation structure
 * 
 * This structure defines the function pointers for implementing the sequence
 * protocol in CPython. It allows custom types to behave like sequences
 * (lists, tuples, strings, etc.) by supporting indexing, slicing, iteration,
 * and in-place operations.
 */
typedef struct
{
  /**
     * @brief Length query function
     * 
     * Implements len(obj) and __len__() protocol.
     * 
     * @return The number of items in the sequence.
     * 
     * @note Required for most sequence operations
     */
  lenfunc sq_length;

  /**
     * @brief Concatenation operation
     * 
     * Implements obj1 + obj2 for sequences.
     * 
     * @param arg1 The left sequence operand
     * @param arg2 The right sequence operand to concatenate
     * @return New reference to concatenated sequence, or NULL on error
     * 
     * @note Maps to __add__() protocol
     * @see sq_inplace_concat for in-place variant
     */
  binaryfunc sq_concat;

  /**
     * @brief Sequence repetition/multiplication operation
     * 
     * Implements obj * n and n * obj for sequences.
     * 
     * @param arg The sequence to repeat
     * @param count Number of times to repeat the sequence
     * @return New reference to repeated sequence, or NULL on error
     * 
     * @note Maps to __mul__() and __rmul__() protocols
     * @see sq_inplace_repeat for in-place variant
     */
  ssizeargfunc sq_repeat;

  /**
     * @brief Item retrieval by index
     * 
     * Implements obj[index] for sequences (subscript access).
     * 
     * @param self The sequence object
     * @param index The index of the item to retrieve (may be negative)
     * @return New reference to the item at index, or NULL on error
     * 
     * @note Maps to __getitem__() with integer index
     * @note Negative indices wrap around from the sequence end
     * @note Called by PySequence_GetItem()
     */
  ssizeargfunc sq_item;

  /**
     * @brief Item assignment/deletion by index
     * 
     * Implements obj[index] = value and del obj[index] for sequences.
     * 
     * @param self The sequence object
     * @param index The index of the item to set
     * @param value The new value, or NULL to delete the item
     * @return 0 on success, -1 on error
     * 
     * @note Maps to __setitem__() and __delitem__() with integer index
     * @note When value is NULL, the item should be deleted
     * @note Negative indices wrap around from the sequence end
     * @note Called by PySequence_SetItem() and PySequence_DelItem()
     */
  ssizeobjargproc sq_ass_item;

  /**
     * @brief Membership test operation
     * 
     * Implements x in obj for sequences (membership testing).
     * 
     * @param self The sequence to search
     * @param arg The value to search for
     * @return 1 if arg is found in self, 0 if not found, -1 on error
     * 
     * @note Maps to __contains__() protocol
     * @note If not implemented, falls back to linear search using sq_item
     * @note Called by PySequence_Contains()
     */
  objobjproc sq_contains;

  /**
     * @brief In-place concatenation operation
     * 
     * Implements obj1 += obj2 for sequences (modifying obj1 in place).
     * 
     * @param self The sequence to modify (may be reallocated)
     * @param arg The sequence to concatenate
     * @return New reference to modified sequence, or NULL on error
     * 
     * @note Maps to __iadd__() protocol
     * @note May return a new object if in-place modification isn't possible
     * @note Called by PySequence_InPlaceConcat()
     * @see sq_concat for the immutable variant
     */
  binaryfunc sq_inplace_concat;

  /**
     * @brief In-place repetition/multiplication operation
     * 
     * Implements obj *= n for sequences (modifying obj in place).
     * 
     * @param self The sequence to repeat and modify in place
     * @param count Number of times to repeat the sequence
     * @return New reference to modified sequence, or NULL on error
     * 
     * @note Maps to __imul__() protocol
     * @note May return a new object if in-place modification isn't possible
     * @note Called by PySequence_InPlaceRepeat()
     * @see sq_repeat for the immutable variant
     */
  ssizeargfunc sq_inplace_repeat;
} __PySequenceMethods;

/** Based on CPython, the idea is to use a PyObject containing type information
 *  while each actual object is explicitly defined.
 */
typedef struct __ESBMC_PyType
{
  const char *tp_name; /* Type name: "module.typename" */
  size_t tp_basicsize; /* Size of instance in bytes */

  // TODO: Extra features (vtables, constructors, members)
  richcmpfunc comparation;
  __PySequenceMethods as_sq;

} PyType;

// TODO: There is no such a thing as a generic type in python.
static PyType __ESBMC_generic_type;

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

// List Type

typedef struct __ESBMC_PyListObj
{
  PyType *type;    /**< &PyListType */
  PyObject *items; /**< Array of PyObject items (SMT infinite array concept) */
  size_t size;     /**< Number of elements currently in use */
} PyListObject;

size_t __python_list_size(__PyObject *obj);
__PyObject *__python_list_concat(__PyObject *obj1, __PyObject *obj2);
__PyObject *__python_list_repeat(__PyObject *self, size_t length);
__PyObject *__python_list_index(__PyObject *self, size_t index);
int __python_list_index_assignment(
  __PyObject *self,
  size_t index,
  __PyObject *value);
int __python_list_contains(__PyObject *self, __PyObject *member);
__PyObject *__python_list_in_concat(__PyObject *self, __PyObject *obj2);
__PyObject *__python_list_in_repeat(__PyObject *self, size_t length);
_Bool __python_list_cmp(__PyObject *a, __PyObject *b, int op);
static PyType __ESBMC_list_type;

int value;
PyType *get_list_type()
{
  if (value)
    return &__ESBMC_list_type;

  __ESBMC_list_type.tp_name = "PyList";
  __ESBMC_list_type.tp_basicsize = 0;

  __ESBMC_list_type.comparation = __python_list_cmp;

  __PySequenceMethods methods;
  methods.sq_length = __python_list_size;
  methods.sq_concat = __python_list_concat;
  methods.sq_repeat = __python_list_repeat;
  methods.sq_item = __python_list_index;
  methods.sq_ass_item = __python_list_index_assignment;
  methods.sq_contains = __python_list_contains;
  methods.sq_inplace_concat = __python_list_in_concat;
  methods.sq_inplace_repeat = __python_list_in_repeat;
  __ESBMC_list_type.as_sq = methods;

  value = 1;
  return &__ESBMC_list_type;
}

PyObject *__ESBMC_create_inf_obj()
{
  return NULL;
};

extern void *__ESBMC_alloca(size_t);

PyListObject *__ESBMC_list_create()
{
  PyListObject *l = __ESBMC_alloca(sizeof(PyListObject));
  l->type = get_list_type();
  l->items = __ESBMC_create_inf_obj();
  l->size = 0;
  return l;
}

size_t __ESBMC_list_size(PyListObject *l)
{
  return get_list_type()->as_sq.sq_length(l);
}

bool __ESBMC_list_push(
  PyListObject *l,
  const void *value,
  size_t type_id,
  size_t type_size)
{
  // TODO: __ESBMC_obj_cpy
  void *copied_value = __ESBMC_alloca(type_size);
  memcpy(copied_value, value, type_size);

  l->items[l->size].value = copied_value;
  l->items[l->size].type_id = type_id;
  l->items[l->size].size = type_size;
  l->size++;

  // TODO: Nondeterministic failure?
  return true;
}

bool __ESBMC_list_push_object(PyListObject *l, PyObject *o)
{
#ifdef __ESBMC_CHECK_OMS
  assert(l != NULL);
  assert(o != NULL);
#endif
  return __ESBMC_list_push(l, o->value, o->type_id, o->size);
}

bool __ESBMC_list_eq(PyListObject *l1, PyListObject *l2)
{
  return get_list_type()->comparation(l1, l2, Py_EQ);
}

PyObject *__ESBMC_list_at(PyListObject *l, size_t index)
{
  return get_list_type()->as_sq.sq_item(l, index);
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
      if (elem->value == item || memcmp(elem->value, item, item_size) == 0)
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

size_t __python_list_size(__PyObject *obj)
{
  PyListObject *l = (PyListObject *)obj;
  return l ? l->size : 0;
}

__PyObject *__python_list_concat(__PyObject *obj1, __PyObject *obj2)
{
  __ESBMC_assert(0, "Not implemented");
  return NULL;
}

__PyObject *__python_list_repeat(__PyObject *self, size_t length)
{
  __ESBMC_assert(0, "Not implemented");
  return NULL;
}

__PyObject *__python_list_index(__PyObject *self, size_t index)
{
  PyListObject *l = (PyListObject *)self;
  __ESBMC_assert(index < l->size, "out-of-bounds read in list");
  return &l->items[index];
}

int __python_list_index_assignment(
  __PyObject *self,
  size_t index,
  __PyObject *value)
{
  __ESBMC_assert(0, "Not implemented");
  return -1;
}

int __python_list_contains(__PyObject *self, __PyObject *member)
{
  __ESBMC_assert(0, "Not implemented");
  return -1;
}

__PyObject *__python_list_in_concat(__PyObject *self, __PyObject *obj2)
{
  __ESBMC_assert(0, "Not implemented");
  return NULL;
}

__PyObject *__python_list_in_repeat(__PyObject *self, size_t length)
{
  __ESBMC_assert(0, "Not implemented");
  return NULL;
}

_Bool __python_list_cmp(__PyObject *lhs, __PyObject *rhs, int op)
{
#ifdef __ESBMC_CHECK_OMS
  __ESBMC_assert(op == Py_EQ, "Only equality supported for now");
#endif
  if (op != Py_EQ)
    return nondet_bool();
  if (!lhs || !rhs)
    return false;
  if (__ESBMC_same_object(lhs, rhs))
    return true;

  PyListObject *l1 = (PyListObject *)lhs;
  PyListObject *l2 = (PyListObject *)rhs;

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
