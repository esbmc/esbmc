#include "cvc_conv.h"

cvc_convt::cvc_convt(bool is_cpp, bool int_encoding, const namespacet &ns)
   : smt_convt(true, int_encoding, ns, is_cpp, false, true, false)
{
  abort();
}

cvc_convt::~cvc_convt()
{
  abort();
}
