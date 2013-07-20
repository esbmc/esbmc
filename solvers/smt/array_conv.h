#ifndef _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_
#define _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_

// Something to abstract the flattening of arrays to QF_BV.

#include "smt_conv.h"

class array_convt : public smt_convt
{
  array_convt(bool enable_cache, bool int_encoding, const namespacet &_ns,
              bool is_cpp, bool tuple_support);
  ~array_convt();
};

#endif /* _ESBMC_SOLVERS_SMT_ARRAY_SMT_CONV_H_ */

