#ifndef PYTHON_LIST_H
#define PYTHON_LIST_H

#include <stddef.h>
#include <stdbool.h>
#include "python_types.h"

// List helpers (defined in list.c)
extern PyListObject *__ESBMC_list_create(void);
extern bool __ESBMC_list_push(
  PyListObject *l,
  const void *value,
  size_t type_id,
  size_t type_size);
extern size_t __ESBMC_list_size(const PyListObject *l);
extern PyObject *__ESBMC_list_at(PyListObject *l, size_t index);

#endif /* PYTHON_LIST_H */
