#include "cnf_conv.h"

cnf_convt::cnf_convt(bool enable_cache, bool int_encoding,
                      const namespacet &_ns, bool is_cpp, bool tuple_support,
                      bool bools_in_arrs, bool can_init_inf_arrs)
  : smt_convt(enable_cache, int_encoding, _ns, is_cpp, tuple_support,
              bools_in_arrs, can_init_inf_arrs)
{
  abort();
}

cnf_convt::~cnf_convt()
{
}
