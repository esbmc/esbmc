#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SORT_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SORT_H_

#include <solvers/smt/smt_sort.h>

// True iff the given type lowers to a tuple sort in the SMT layer:
// struct (incl. C++ class data), pointer (as the (object, offset)
// pair lowered via pointer_struct), code (function pointer payload),
// or complex (the (real, imag) pair).
inline bool is_tuple_ast_type(const type2tc &t)
{
  return is_struct_type(t) || is_pointer_type(t) || is_code_type(t) ||
         is_complex_type(t);
}

inline bool is_tuple_ast_type(const expr2tc &e)
{
  return is_tuple_ast_type(e->type);
}

inline bool is_tuple_array_ast_type(const type2tc &t)
{
  if (!is_array_type(t))
    return false;

  const array_type2t &arr_type = to_array_type(t);
  type2tc range = arr_type.subtype;
  while (is_array_type(range))
    range = to_array_type(range).subtype;

  return is_tuple_ast_type(range);
}

#endif
