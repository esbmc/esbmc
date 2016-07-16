#include "parseoptions.h"
#include <irep2.h>
#include <solvers/smt/smt_conv.h>
#include <langapi/mode.h>

#include <boost/python.hpp>
using namespace boost::python;

void build_bigint_python_class();
void build_base_expr2t_python_class();
void build_base_type2t_python_class();
void build_type2t_container_converters();
void build_expr2t_container_converters();
void build_dstring_python_class();
void build_smt_conv_python_class();

class dummy_expr_class { };
class dummy_type_class { };

// Prevent more than one instance per process
static bool python_module_engaged = false;
// Parseoptions instance representing an esbmc process
static cbmc_parseoptionst *po = NULL;
static namespacet *ns = NULL;

static boost::python::object
init_esbmc_process(boost::python::object o)
{
  using namespace boost::python;
  std::vector<std::string> str_list;
  const char **argv;
  unsigned int argc, i;
  int result;

  // Arguments: list of options that we would otherwise provide on the ESBMC
  // command line. Convert these to argc/argv for parseoptions, create
  // parseoptions object, perform initial processing, and return handles to
  // useful structures.

  // Extract list from object; provoke exception if needs be.
  list l = extract<list>(o);

  // Apparently no good way to iterate over list
  while (len(l) > 0) {
    object s = l.pop();
    str_list.push_back(extract<std::string>(s));
  }

  // Convert the list of C++ lists to C argc / argv.
  argc = str_list.size();
  argc += 1; // Extra option we add.
  argv = (const char**)malloc(sizeof(const char *) * argc);
  i = 0;
  for (const std::string &s : str_list)
    argv[i++] = s.data();

  // Add skip-bmc option: causes all usual processing to happen, but we bail
  // out of parseoptions at the point where we would usually start BMC.
  argv[i++] = "--skip-bmc";

  // Init esbmc Stuff
  python_module_engaged = true;
  po = new cbmc_parseoptionst(argc, argv);
  free(argv);

  // Perform initial processing
  result = po->doit();

  // Assuming we didn't abort; if there's an error, return None. Otherwise
  // construct a tuple of useful handles.
  if (result != 0) {
    delete po;
    python_module_engaged = false;
    return object();
  }

  ns = new namespacet(po->context);

  // Convert return values to python objects (TM).
  object nso(ns);
  // Config options are global. Woo.
  object opts(&config.options);

  return make_tuple(nso, opts);
}

static void
kill_esbmc_process(void)
{
  if (!python_module_engaged)
    // Nope
    return;

  assert(po != NULL && ns != NULL);

  // It's the users problem if they haven't actually cleaned up their python
  // references.
  delete ns;
  ns = NULL;
  delete po;
  po = NULL;
  python_module_engaged = false;

  return;
}

BOOST_PYTHON_MODULE(esbmc)
{
  // Register process init and sort-of deconstruction.
  def("init_esbmc_process", &init_esbmc_process);
  def("kill_esbmc_process", &kill_esbmc_process);

  // Use boost preprocessing iteration to enumerate all irep classes and
  // register them into python. In the future this should be done via types
  // so that it can actually be typechecked, but that will require:
  //  * A boost set of ireps to exist
  //  * expr_id's to be registered into a template like irep_methods2.

  // Namespace into types and exprs.
  {
    scope types = class_<dummy_type_class>("type");

    build_base_type2t_python_class();
#define _ESBMC_IREP2_MPL_TYPE_SET(r, data, elem) BOOST_PP_CAT(elem,_type2t)::build_python_class(type2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(_ESBMC_IREP2_MPL_TYPE_SET, foo, ESBMC_LIST_OF_TYPES)

    build_type2t_container_converters();
  }

  {
    scope types = class_<dummy_expr_class>("expr");

    build_base_expr2t_python_class();
#define _ESBMC_EXPR2_MPL_EXPR_SET(r, data, elem) BOOST_PP_CAT(elem,2t)::build_python_class(expr2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(_ESBMC_EXPR2_MPL_EXPR_SET, foo, ESBMC_LIST_OF_EXPRS)

    build_expr2t_container_converters();
  }

  // Register BigInt globally
  build_bigint_python_class();
  build_dstring_python_class();

  // Alas, we need to pass handles to optionst around, and namespace. User
  // should be able to extract them from whatever execution context they
  // generate.
  opaque<optionst>();
  opaque<namespacet>();

  // Build smt solver related stuff
  build_smt_conv_python_class();
}

// Include these other things that are special to the esbmc binary:

const mode_table_et mode_table[] =
{
  LANGAPI_HAVE_MODE_C,
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CLANG_C,
#endif
  LANGAPI_HAVE_MODE_CPP,
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CLANG_CPP,
#endif
  LANGAPI_HAVE_MODE_END
};

extern "C" uint8_t buildidstring_buf[1];
uint8_t *version_string = buildidstring_buf;
