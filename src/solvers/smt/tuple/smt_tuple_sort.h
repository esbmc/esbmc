#ifndef SOLVERS_SMT_TUPLE_SMT_TUPLE_SORT_H_
#define SOLVERS_SMT_TUPLE_SMT_TUPLE_SORT_H_

#include <solvers/smt/smt_sort.h>

class tuple_smt_sort;
typedef const tuple_smt_sort *tuple_smt_sortt;

/** Storage for flattened tuple sorts.
 *  When flattening tuples (and arrays of them) down to SMT, we need to store
 *  additional type data. This sort is used in tuple code to record that data.
 *  @see smt_tuple.cpp */
class tuple_smt_sort : public smt_sort
{
public:
  /** Actual type (struct or array of structs) of the tuple that's been
   * flattened */
  const type2tc thetype;

  tuple_smt_sort(const type2tc &type) : smt_sort(SMT_SORT_STRUCT), thetype(type)
  {
  }

  tuple_smt_sort(
    const type2tc &type,
    unsigned long range_width,
    unsigned long dom_width)
    : smt_sort(SMT_SORT_ARRAY, range_width, dom_width), thetype(type)
  {
  }

  ~tuple_smt_sort() override = default;
};

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

inline tuple_smt_sortt to_tuple_sort(smt_sortt a)
{
  tuple_smt_sortt ta = dynamic_cast<tuple_smt_sortt>(a);
  assert(ta != nullptr && "Tuple AST mismatch");
  return ta;
}

#endif
