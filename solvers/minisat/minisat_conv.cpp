#include "minisat_conv.h"

minisat_convt::minisat_convt(bool int_encoding, const namespacet &_ns,
                             bool is_cpp, const optionst &_opts __attribute__((unused))) : smt_convt(true, int_encoding, _ns, is_cpp, false, true, true)
{
  abort();
}

minisat_convt::~minisat_convt(void)
{
}
