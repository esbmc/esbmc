#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <solvers/smt/smt_conv.h>
#include <boost/python/class.hpp>

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_ast)
BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_sort)

class dummy_solver_class;

void
build_smt_conv_python_class(void)
{
  using namespace boost::python;

  scope solve = class_<dummy_solver_class>("solve");

  // There are no circumstances where python users should be inspecting the
  // internals of SMT handles. They're handles. In terms of lifetime, the smt
  // handles get destroyed when the smt_convt gets destroyed. Now, it would
  // be illegal to pass a handle from one smt_convt to another smt_convt; in
  // the same way, if your smt_convt has gone out of scope and been destroyed,
  // you don't have anywhere to legally pass these opqaue pointers.
  //
  // That rather ignores the matter of push/popping scopes. User is on their
  // own there.
  opaque<smt_ast>();
  opaque<smt_sort>();

  // Register generic smt_convt facilities: only allow the python user to do
  // expression conversion. Any new smt_convt implementation should be done
  // in C++ for example.
  (void);

  // Register list of available solvers, as factory functions.
  for (unsigned int i = 0; i < esbmc_num_solvers; i++) {
  }
}
#endif
