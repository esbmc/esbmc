#include <stdint.h> // SIZE_MAX
#include "list.h"

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
