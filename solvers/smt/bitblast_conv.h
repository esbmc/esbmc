#ifndef _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_
#define _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_

#include "smt_conv.h"

class bitblast_convt : public virtual smt_convt
{
public:
  bitblast_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
                 bool is_cpp, bool tuple_support, bool bools_in_arrs,
                 bool can_init_inf_arrs);
  ~bitblast_convt();
};

#endif /* _ESBMC_SOLVERS_SMT_BITBLAST_CONV_H_ */
