#ifndef _ESBMC_SOLVERS_SMT_CNF_CONV_H_
#define _ESBMC_SOLVERS_SMT_CNF_CONV_H_

#include <solvers/smt/smt_conv.h>
#include <bitblast_conv.h>
#include <cnf_iface.h>

class cnf_convt : public sat_iface
{
public:
  cnf_convt(cnf_iface *cnf_api);
  ~cnf_convt();

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

  cnf_iface *cnf_api;
};

#endif /* _ESBMC_SOLVERS_SMT_CNF_CONV_H_ */
