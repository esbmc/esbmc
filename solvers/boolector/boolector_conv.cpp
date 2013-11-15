#include <irep2.h>
#include <namespace.h>

#include <solvers/smt/smt_conv.h>

#include <boolector.h>

smt_convt *
create_new_boolector_solver(bool is_cpp, bool int_encoding,
                            const namespacet &ns)
{
  abort();
#if 0
    return new boolector_convt(int_encoding, ns, is_cpp);
#endif
}

