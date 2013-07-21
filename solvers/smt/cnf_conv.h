#ifndef _ESBMC_SOLVERS_SMT_CNF_CONV_H_
#define _ESBMC_SOLVERS_SMT_CNF_CONV_H_

#include "smt_conv.h"

template <class subclass>
class cnf_convt : public virtual subclass
{
public:
  cnf_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
                 bool is_cpp, bool tuple_support, bool bools_in_arrs,
                 bool can_init_inf_arrs);
  ~cnf_convt();

  // The API we require:
  virtual void setto(literalt a, bool val) = 0;
  virtual void lcnf(const bvt &bv) = 0;

  // The API we're implementing: all reducing to cnf(), eventually.
  virtual literalt lnot(literalt a);
  virtual literalt lselect(literalt a, literalt b, literalt c);
  virtual literalt lequal(literalt a, literalt b);
  virtual literalt limplies(literalt a, literalt b);
  virtual literalt lxor(literalt a, literalt b);
  virtual literalt lor(literalt a, literalt b);
  virtual literalt land(literalt a, literalt b);
  virtual void gate_xor(literalt a, literalt b, literalt o);
  virtual void gate_or(literalt a, literalt b, literalt o);
  virtual void gate_and(literalt a, literalt b, literalt o);
  virtual void set_equal(literalt a, literalt b);
};

// And because this is a template...
#include "cnf_conv.cpp"

#endif /* _ESBMC_SOLVERS_SMT_CNF_CONV_H_ */
