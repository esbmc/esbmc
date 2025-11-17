#pragma once

#include <util/expr.h>

namespace python_char_utils
{
// Attempt to extract a single character expression from `expr`.
// When allow_void_pointer is true, treat void* values (empty subtype pointers)
// as pointers to a single-character buffer.
bool try_get_single_char_expr(
  const exprt &expr,
  exprt &char_expr,
  bool allow_void_pointer = false);

// Returns the character value converted to signed 8-bit integer.
// If the expression is not a single character, returns nil_exprt.
exprt get_char_value_as_int(const exprt &expr, bool allow_void_pointer = false);

} // namespace python_char_utils
