#include "smtlib_conv.h"

smtlib_convt::smtlib_convt(bool int_encoding, const namespacet &_ns,
                           bool is_cpp)
  : smt_convt(false, int_encoding, _ns, is_cpp, false)
{
}

smtlib_convt::~smtlib_convt()
{
}
