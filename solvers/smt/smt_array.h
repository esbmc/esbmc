#include "smt_conv.h"

// Interface definition for array manipulation

class array_iface
{
public:
  virtual smt_astt mk_array_symbol(const std::string &name, smt_sortt sort) = 0;
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
  virtual const smt_ast *convert_array_of(const expr2tc &init_val,
                                           unsigned long domain_width) = 0;

  static const smt_ast *default_convert_array_of(const expr2tc &init_val,
                                          unsigned long domain_width,
                                          smt_convt *ctx);

  // And everything else goes through the ast methods!
};
