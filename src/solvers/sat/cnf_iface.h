#ifndef _ESBMC_SOLVERS_SAT_CNF_IFACE_H_
#define _ESBMC_SOLVERS_SAT_CNF_IFACE_H_

class cnf_iface
{
public:
  virtual void setto(literalt a, bool val) = 0;
  virtual void lcnf(const bvt &bv) = 0;
};

#endif /* _ESBMC_SOLVERS_SAT_CNF_IFACE_H_ */
