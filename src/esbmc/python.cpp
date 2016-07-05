#include "parseoptions.h"
#include <irep2.h>
#include <solvers/smt/smt_conv.h>
#include <langapi/mode.h>

#include <boost/python.hpp>
using namespace boost::python;

void build_bigint_python_class();

class dummy_expr_class { };
class dummy_type_class { };

BOOST_PYTHON_MODULE(esbmc)
{
  // Use boost preprocessing iteration to enumerate all irep classes and
  // register them into python. In the future this should be done via types
  // so that it can actually be typechecked, but that will require:
  //  * A boost set of ireps to exist
  //  * expr_id's to be registered into a template like irep_methods2.

  // Namespace into types and exprs.
  {
    scope types = class_<dummy_type_class>("type");

#define _ESBMC_IREP2_MPL_TYPE_SET(r, data, elem) BOOST_PP_CAT(elem,_type2t)::build_python_class(type2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(_ESBMC_IREP2_MPL_TYPE_SET, foo, ESBMC_LIST_OF_TYPES)
  }

  {
    scope types = class_<dummy_expr_class>("expr");

#define _ESBMC_EXPR2_MPL_EXPR_SET(r, data, elem) BOOST_PP_CAT(elem,2t)::build_python_class(expr2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(_ESBMC_EXPR2_MPL_EXPR_SET, foo, ESBMC_LIST_OF_EXPRS)
  }

  // Register BigInt globally
  build_bigint_python_class();
}

// Include these other things that are special to the esbmc binary:

const mode_table_et mode_table[] =
{
  LANGAPI_HAVE_MODE_C,
  LANGAPI_HAVE_MODE_CLANG_C,
  LANGAPI_HAVE_MODE_CPP,
  LANGAPI_HAVE_MODE_CLANG_CPP,
  LANGAPI_HAVE_MODE_END
};

extern "C" uint8_t buildidstring_buf[1];
uint8_t *version_string = buildidstring_buf;
