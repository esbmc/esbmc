#include "parseoptions.h"
#include <irep2.h>
#include <solvers/smt/smt_conv.h>
#include <langapi/mode.h>
#include <goto-programs/goto_functions.h>

#include <boost/python.hpp>
using namespace boost::python;

void dereference_handlers_init(void);
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
// Type pool needs to live as long as the process.
static type_poolt *tp = NULL;

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

  // Init esbmc Stuff. First the static order initialization fiasco.
  tp = new type_poolt(true);
  type_pool = *tp;
  init_expr_constants();
  dereference_handlers_init();

  python_module_engaged = true;
  po = new cbmc_parseoptionst(argc, argv);
  free(argv);

  // Perform initial processing
  result = po->doit();

  // Assuming we didn't abort; if there's an error, return None. Otherwise
  // construct a tuple of useful handles.
  if (result != 0) {
    delete po;
    delete tp;
    python_module_engaged = false;
    return object();
  }

  ns = new namespacet(po->context);

  // Convert return values to python objects (TM). Wrap into a PyObject, stuff
  // in handle, transfer to object. Unclear if there's a supported way of
  // doing this with opaque pointers in the API: I get the impression that
  // they're only supposed to be single return values.
  auto converter1 = converter::registry::lookup(type_id<namespacet*>());
  handle<> nsh(converter1.to_python(&ns));
  object nso(nsh);
  // Config options are global. Woo.
  auto converter2 = converter::registry::lookup(type_id<optionst*>());
  auto opt_ptr = &config.options;
  handle<> optsh(converter2.to_python(&opt_ptr));
  object opts(optsh);
  // Goto functions handle comes from inside the parseoptions object.
  auto converter3 = converter::registry::lookup(type_id<goto_functionst*>());
  auto func_ptr = &po->goto_functions;
  handle<> funch(converter3.to_python(&func_ptr));
  object funcs(funch);

  return make_tuple(nso, opts, funcs);
}

static void
kill_esbmc_process(void)
{
  if (!python_module_engaged)
    // Nope
    return;

  assert(po != NULL && ns != NULL && tp != NULL);

  // It's the users problem if they haven't actually cleaned up their python
  // references.
  delete ns;
  ns = NULL;
  delete po;
  po = NULL;
  delete tp;
  tp = NULL;
  python_module_engaged = false;

  return;
}

BOOST_PYTHON_MODULE(esbmc)
{
  // This is essentially the entry point for the esbmc shared object.
  // Workarounds for the static order initialization are in init_esbmc_process
  // due to some annoyance with object lifetime.

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

  // Alas, we need to pass handles to optionst, namespace, goto funcs around.
  // User should be able to extract them from whatever execution context they
  // generate.
  opaque<optionst>();
  opaque<namespacet>();
  opaque<goto_functionst>();

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
