#include "array_conv.h"

array_convt::array_convt(bool enable_cache, bool int_encoding,
                         const namespacet &_ns, bool is_cpp, bool tuple_support)
  // Declare that we can put bools in arrays, and init unbounded arrays
  : smt_convt(enable_cache, int_encoding, _ns, is_cpp, tuple_support, false,
              true)
{
  abort();
}

array_convt::~array_convt()
{
}
