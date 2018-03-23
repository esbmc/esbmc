#ifndef _ESBMC_SOLVERS_SAT_SAT_IFACE_H_
#define _ESBMC_SOLVERS_SAT_SAT_IFACE_H_

// An interface for defining a SAT interface within ESBMC, as used by the
// SAT bitblaster. I anticipate that nothing else actually needs to use this
// interface, except perhaps sat solvers that have non-cnf inputs.

class sat_iface
{
public:
  virtual void lcnf(const bvt &bv) = 0;
  virtual literalt lnot(literalt a) = 0;
  virtual literalt lselect(literalt a, literalt b, literalt c) = 0;
  virtual literalt lequal(literalt a, literalt b) = 0;
  virtual literalt limplies(literalt a, literalt b) = 0;
  virtual literalt lxor(literalt a, literalt b) = 0;
  virtual literalt land(literalt a, literalt b) = 0;
  virtual literalt lor(literalt a, literalt b) = 0;
  virtual void set_equal(literalt a, literalt b) = 0;
  virtual void assert_lit(const literalt &a) = 0;
  virtual tvt l_get(const literalt &a) = 0;
  virtual literalt new_variable() = 0;
};

#endif /* _ESBMC_SOLVERS_SAT_SAT_IFACE_H_ */
