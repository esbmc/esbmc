#include "metasmt_conv.h"

metasmt_convt::metasmt_convt(bool int_encoding, bool is_cpp,
                             const namespacet &ns)
  : smt_convt(false, int_encoding, ns, is_cpp, false)
{
}

metasmt_convt::~metasmt_convt()
{
}
