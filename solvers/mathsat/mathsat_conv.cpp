#include "mathsat_conv.h"

mathsat_convt::mathsat_convt(bool is_cpp, bool int_encoding,
                             const namespacet &ns)
  : smt_convt(true, int_encoding, ns, is_cpp, false, true, true)
{
  abort();
}

mathsat_convt::~mathsat_convt(void)
{
}
