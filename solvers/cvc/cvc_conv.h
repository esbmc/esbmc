#ifndef _ESBMC_SOLVERS_CVC_CVC_CONV_H_
#define _ESBMC_SOLVERS_CVC_CVC_CONV_H_

#include <solvers/smt/smt_conv.h>

class cvc_convt : public smt_convt
{
public:
  cvc_convt(bool is_cpp, bool int_encoding, const namespacet &ns);
  ~cvc_convt();
};

#endif /* _ESBMC_SOLVERS_CVC_CVC_CONV_H_ */
