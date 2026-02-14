#ifndef PYTHON_TYPES_H
#define PYTHON_TYPES_H

#include <stddef.h>

// Bounded string length used by Python frontend/runtime helpers.
// Keeps symbolic loops finite while still respecting '\0' within the bound.
#ifndef ESBMC_PY_STRNLEN_BOUND
#  define ESBMC_PY_STRNLEN_BOUND 256
#endif

#ifndef ESBMC_PY_STR_TYPE_ID
#  define ESBMC_PY_STR_TYPE_ID ((size_t)0x826e83195d0d60f0ULL)
#endif

/**
 * @brief Type object representation for Python-like types.
 */
typedef struct __ESBMC_PyType
{
  const char *tp_name; /* Type name: "module.typename" */
  size_t tp_basicsize; /* Size of instance in bytes */
  // TODO: Extra features (vtables, constructors, members)
} PyType;

/**
 * @brief Minimal representation of a Python-like object.
 *
 * In CPython, PyObject includes only a pointer to its type object. Most
 * user-defined types embed this as their header, allowing any instance to
 * be safely cast to PyObject* for type inspection.
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

#endif /* PYTHON_TYPES_H */
