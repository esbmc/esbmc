#ifndef _ESBMC_SOLVERS_SMT_CNF_CONV_H_
#define _ESBMC_SOLVERS_SMT_CNF_CONV_H_

#include "smt_conv.h"

class cnf_convt : public virtual smt_convt
{
  cnf_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
                 bool is_cpp, bool tuple_support, bool bools_in_arrs,
                 bool can_init_inf_arrs);
  ~cnf_convt();
};

#endif /* _ESBMC_SOLVERS_SMT_CNF_CONV_H_ */
