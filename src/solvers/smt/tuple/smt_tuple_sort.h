#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SORT_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SORT_H_

#include <solvers/smt/smt_sort.h>

#define is_tuple_ast_type(x) (is_structure_type(x) || is_pointer_type(x))

inline bool is_tuple_array_ast_type(const type2tc &t)
{
  if(!is_array_type(t))
    return false;

  const array_type2t &arr_type = to_array_type(t);
  type2tc range = arr_type.subtype;
  while(is_array_type(range))
    range = to_array_type(range).subtype;

  return is_tuple_ast_type(range);
}

#endif
