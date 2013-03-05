#ifndef _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H
#define _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H

#include <irep.h>
#include <solvers/prop/prop_conv.h>
#include <solvers/smt/smt_conv.h>

class smtlib_convt : public smt_convt {
  smtlib_convt(bool int_encoding, const namespacet &_ns, bool is_cpp);
  ~smtlib_convt();
};

#endif /* _ESBMC_SOLVERS_SMTLIB_SMTLIB_CONV_H */
