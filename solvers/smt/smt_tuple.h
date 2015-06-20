#ifndef _ESBMC_SOLVERS_SMT_SMT_TUPLE_H_
#define _ESBMC_SOLVERS_SMT_SMT_TUPLE_H_

#include "smt_conv.h"

// Abstract class defining the interface required for creating tuples.
class tuple_iface {
public:
  /** Create a sort representing a struct. i.e., a tuple. Ideally this should
   *  actually be part of the overridden tuple api, but due to history it isn't
   *  yet. If solvers don't support tuples, implement this to abort.
   *  @param type The struct type to create a tuple representation of.
   *  @return The tuple representation of the type, wrapped in an smt_sort. */
  virtual smt_sortt mk_struct_sort(const type2tc &type) = 0;

  /** Create a new tuple from a struct definition.
   *  @param structdef A constant_struct2tc, describing all the members of the
   *         tuple to create.
   *  @return AST representing the created tuple */
  virtual smt_astt tuple_create(const expr2tc &structdef) = 0;

  /** Create a fresh tuple, with freely valued fields.
   *  @param s Sort of the tuple to create
   *  @return AST representing the created tuple */
  virtual smt_astt tuple_fresh(smt_sortt s, std::string name = "") = 0;

  // XXX XXX XXX docs gap
  virtual smt_astt tuple_array_create(const type2tc &array_type,
                              smt_astt *inputargs,
                              bool const_array,
                              smt_sortt domain) = 0;

  /** Create a potentially /large/ array of tuples. This is called when we
   *  encounter an array_of operation, with a very large array size, of tuple
   *  sort.
   *  @param Expression of tuple value to populate this array with.
   *  @param domain_width The size of array to create, in domain bits.
   *  @return An AST representing an array of the tuple value, init_value. */
  virtual smt_astt tuple_array_of(const expr2tc &init_value,
                                        unsigned long domain_width) = 0;

  /** Convert a symbol to a tuple_smt_ast */
  virtual smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) = 0;

  /** Like mk_tuple_symbol, but for arrays */
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr) = 0;

  /** Extract the assignment to a tuple-typed symbol from the SMT solvers
   *  model */
  virtual expr2tc tuple_get(const expr2tc &expr) = 0;

  virtual void add_tuple_constraints_for_solving() = 0;
  virtual void push_tuple_ctx() = 0;
  virtual void pop_tuple_ctx() = 0;
};

#endif /* _ESBMC_SOLVERS_SMT_SMT_TUPLE_H_ */
