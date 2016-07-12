#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <solvers/smt/smt_conv.h>
#include <boost/python/class.hpp>

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_ast)
BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_sort)

void
build_smt_conv_python_class(void)
{
  using namespace boost::python;

  opaque<smt_ast>();
  opaque<smt_sort>();
}
#endif
