#ifndef _ESBMC_SOLVERS_SMT_SMT_ARRAY_H_
#define _ESBMC_SOLVERS_SMT_SMT_ARRAY_H_

#include <solvers/smt/smt_conv.h>

// Interface definition for array manipulation

class array_iface
{
public:
  // This constructor makes this not qualify as an abstract interface according
  // to strict definitions.
  array_iface(bool _b, bool inf) : supports_bools_in_arrays(_b),
                                   can_init_infinite_arrays(inf) { }

  virtual smt_astt mk_array_symbol(const std::string &name, smt_sortt sort,
                                   smt_sortt subtype) = 0;

  /** Extract an element from the model of an array, at an explicit index.
   *  @param array AST representing the array we are extracting from
   *  @param index The index of the element we wish to expect
   *  @param subtype The type of the element we are extracting, i.e. array range
   *  @return Expression representation of the element */
  virtual expr2tc get_array_elem(smt_astt a, uint64_t idx,
                                 const type2tc &subtype) = 0;

  /** Create an array with a single initializer. This may be a small, fixed
   *  size array, or it may be a nondeterministically sized array with a
   *  word-sized domain. Default implementation is to repeatedly store into
   *  the array for as many elements as necessary; subclassing class should
   *  override if it has a more efficient method.
   *  Nondeterministically sized memory with an initializer is very rare;
   *  the only real users of this are fixed sized (but large) static arrays
   *  that are zero initialized, or some infinite-domain modelling arrays
   *  used in ESBMC.
   *  @param init_val The value to initialize each element with.
   *  @param domain_width The size of the array to create, in domain bits.
   *  @return An AST representing the created constant array. */
  virtual const smt_ast *convert_array_of(smt_astt init_val,
                                           unsigned long domain_width) = 0;

  const smt_ast *default_convert_array_of(smt_astt init_val,
                                          unsigned long domain_width,
                                          smt_convt *ctx);

  virtual void add_array_constraints_for_solving() = 0;

  virtual void push_array_ctx() = 0;
  virtual void pop_array_ctx() = 0;

  // And everything else goes through the ast methods!

  // Small piece of internal munging:
  bool supports_bools_in_arrays;
  bool can_init_infinite_arrays;
};

#endif /* _ESBMC_SOLVERS_SMT_SMT_ARRAY_H_ */
