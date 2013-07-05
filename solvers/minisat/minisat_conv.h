#ifndef _ESBMC_SOLVERS_SMTLIB_CONV_H_
#define _ESBMC_SOLVERS_SMTLIB_CONV_H_

#include <solvers/smt/smt_conv.h>

class minisat_convt : public smt_convt {
  minisat_convt(bool int_encoding, const namespacet &_ns, bool is_cpp,
                const optionst &opts);
  ~minisat_convt();
};

#endif /* _ESBMC_SOLVERS_SMTLIB_CONV_H_ */
