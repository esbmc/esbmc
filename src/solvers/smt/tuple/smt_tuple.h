#ifndef _ESBMC_SOLVERS_SMT_TUPLE_SMT_TUPLE_H_
#define _ESBMC_SOLVERS_SMT_TUPLE_SMT_TUPLE_H_

#include <solvers/smt/smt_ast.h>
#include <solvers/smt/tuple/smt_tuple_sort.h>

/** Reject a tuple field index that the AST cannot hold.
 *
 *  smt_solver_baset::convert_member and the with_id case of convert_ast take
 *  the index from the *expression's* struct type, while the AST being indexed
 *  carries the sort it was built from. Where a frontend leaves the two
 *  disagreeing, an unchecked index runs off the end of the flattener's vector
 *  -- a read in project(), a write in update() -- and the process dies far
 *  from the cause. Both flatteners route every index through here so the
 *  failure names the tuple, the field and the two sizes instead.
 *
 *  @param idx Field index, as computed from the expression's type.
 *  @param size Number of fields the AST actually holds.
 *  @param tuple_type The sort's struct/union/complex type, for the message. */
void check_tuple_field(
  unsigned int idx,
  std::size_t size,
  const type2tc &tuple_type);

// Abstract class defining the interface required for creating tuples.
class tuple_iface
{
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
  virtual smt_astt tuple_array_create(
    const type2tc &array_type,
    smt_astt *inputargs,
    bool const_array,
    smt_sortt domain) = 0;

  /** Create a potentially /large/ array of tuples. This is called when we
   *  encounter an array_of operation, with a very large array size, of tuple
   *  sort.
   *  @param Expression of tuple value to populate this array with.
   *  @param domain_width The size of array to create, in domain bits.
   *  @return An AST representing an array of the tuple value, init_value. */
  virtual smt_astt
  tuple_array_of(const expr2tc &init_value, unsigned long domain_width) = 0;

  /** Convert a symbol to a tuple_smt_ast */
  virtual smt_astt mk_tuple_symbol(const std::string &name, smt_sortt s) = 0;

  /** Like mk_tuple_symbol, but for arrays */
  virtual smt_astt mk_tuple_array_symbol(const expr2tc &expr) = 0;

  /** Extract the assignment to a tuple-typed symbol from the SMT solvers
   *  model */
  virtual expr2tc tuple_get(const expr2tc &expr) = 0;
  virtual expr2tc tuple_get(const type2tc &type, smt_astt a) = 0;

  virtual expr2tc tuple_get_array_elem(
    smt_astt array,
    uint64_t index,
    const type2tc &subtype) = 0;

  virtual void add_tuple_constraints_for_solving(){};
  virtual void push_tuple_ctx(){};
  virtual void pop_tuple_ctx(){};

  virtual ~tuple_iface() = default;
};

#endif /* _ESBMC_SOLVERS_SMT_TUPLE_SMT_TUPLE_H_ */
