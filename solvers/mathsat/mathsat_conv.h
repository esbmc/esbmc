#ifndef _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_
#define _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_

#include <solvers/smt/smt_conv.h>

class mathsat_convt : public smt_convt
{
  mathsat_convt(bool is_cpp, bool int_encoding, const namespacet &ns);
  ~mathsat_convt(void);
};

#endif /* _ESBMC_SOLVERS_MATHSAT_MATHSAT_CONV_H_ */
