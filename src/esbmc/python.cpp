#include <esbmc/esbmc_parseoptions.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <solvers/smt/smt_conv.h>
#include <langapi/mode.h>
#include <goto-programs/goto_functions.h>
#include <boost/python.hpp>
#include <boost/python/object/find_instance.hpp>
#include <util/bp_converter.h>

using namespace boost::python;

class location
{
public:
  location(const locationt &loc)
  {
    file = loc.get_file();
    line = string2integer(loc.get_line().as_string()).to_ulong();
    column = string2integer(loc.get_column().as_string()).to_ulong();
    function = loc.get_function();
  }

  location(const irep_idt &_f, unsigned int l, unsigned int c,
      const irep_idt &_func) : file(_f), line(l), column(c), function(_func)
  { }

  static location from_locationt(const locationt &loc) {
    return location(loc);
  }

  irep_idt file;
  unsigned int line;
  unsigned int column;
  irep_idt function;
};

void build_bigint_python_class();
void build_guard_python_class();
void build_base_expr2t_python_class();
void build_base_type2t_python_class();
void build_type2t_container_converters();
void build_expr2t_container_converters();
void build_dstring_python_class();
void build_smt_conv_python_class();
void build_goto_func_class();
void build_fixedbv_python_class();
void build_goto_symex_classes();
void build_equation_class();
void build_value_set_classes();

class dummy_expr_class { };
class dummy_type_class { };
class dummy_symex_class { };

// Prevent more than one instance per process
static bool python_module_engaged = false;
// Parseoptions instance representing an esbmc process
static cbmc_parseoptionst *po = NULL;
namespacet *pythonctx_ns = NULL;
// Type pool needs to live as long as the process.
static type_poolt *tp = NULL;

dict *type_to_downcast = NULL;
dict *expr_to_downcast = NULL;

static void
its_a_trap()
{
  __asm__("int $3");
  __asm__("int $3"); // Repeat for gdb to latch onto a line number
}

template <typename T>
class migrate_func;

template<>
class migrate_func<type2tc>
{
public:
  static void *rvalue_cvt(const typet *type, type2tc *out)
  {
    new (out) type2tc();
    migrate_type(*type, *out);
    return (void*)out;
  }

  static void *lvalue_cvt(const typet *foo)
  {
    return const_cast<void *>(reinterpret_cast<const void*>(foo));
  }
};

template<>
class migrate_func<expr2tc>
{
public:
  static void *rvalue_cvt(const exprt *expr, expr2tc *out)
  {
    new (out) expr2tc();
    migrate_expr(*expr, *out);
    return (void*)out;
  }

  static void *lvalue_cvt(const exprt *foo)
  {
    return const_cast<void *>(reinterpret_cast<const void*>(foo));
  }
};

template<>
class migrate_func<typet>
{
public:
  static void *rvalue_cvt(const type2tc *type, typet *out)
  {
    new (out) typet();
    *out = migrate_type_back(*type);
    return (void*)out;
  }

  static void *lvalue_cvt(const type2tc *foo)
  {
    return const_cast<void *>(reinterpret_cast<const void*>(foo));
  }
};

template<>
class migrate_func<exprt>
{
public:
  static void *rvalue_cvt(const expr2tc *expr, exprt *out)
  {
    new (out) exprt();
    *out = migrate_expr_back(*expr);
    return (void*)out;
  }

  static void *lvalue_cvt(const expr2tc *foo)
  {
    return const_cast<void *>(reinterpret_cast<const void*>(foo));
  }

};

void
register_oldrep_to_newrep()
{
  esbmc_python_cvt<type2tc, typet, false, true, false, migrate_func<type2tc> >();
  return;
}

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

  for (unsigned int i = 0; i < len(l); i++)
    str_list.push_back(extract<std::string>(l[i]));

  // Convert the list of C++ lists to C argc / argv.
  argc = str_list.size();
  argc += 2; // Extra options we add.
  argv = (const char**)malloc(sizeof(const char *) * argc);
  i = 0;
  argv[i++] = "esbmc"; // ESBMC expects program path to be first arg
  for (const std::string &s : str_list)
    argv[i++] = s.data();

  // Add skip-bmc option: causes all usual processing to happen, but we bail
  // out of parseoptions at the point where we would usually start BMC.
  argv[i++] = "--skip-bmc";

  // Init esbmc Stuff. First the static order initialization fiasco.
  tp = new type_poolt(true);
  type_pool = *tp;

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

  pythonctx_ns = new namespacet(po->context);

  // Convert return values to python objects (TM). Wrap into a PyObject, stuff
  // in handle, transfer to object.
  object nso(pythonctx_ns);
  // Config options are global. Woo.
  auto opt_ptr = &config.options;
  object opts(opt_ptr);
  // Emit internal reference to parseoptions object. It's the python users
  // problem if it calls kill_esbmc_process and then touches references to
  // this.
  reference_existing_object::apply<cbmc_parseoptionst*>::type po_cvt;
  PyObject *pop = po_cvt(po);
  handle<> poh(pop);
  object po_obj(poh);

  return make_tuple(nso, opts, po_obj);
}

static void
kill_esbmc_process(void)
{
  if (!python_module_engaged)
    // Nope
    return;

  assert(po != NULL && pythonctx_ns != NULL && tp != NULL);

  // It's the users problem if they haven't actually cleaned up their python
  // references.
  delete pythonctx_ns;
  pythonctx_ns = NULL;
  delete po;
  po = NULL;
  delete tp;
  tp = NULL;
  python_module_engaged = false;

  return;
}

// For numerous reasons we want facilities to downcast a type2tc or expr2tc
// to the corresponding something2tc class, allowing python to access the
// contents. However, to_##thetype##_type etc are a) overloaded, and b) don't
// return something2tc's. And we can't register the something2tc constructor
// as a simple function. So we get this:
template <typename Result, typename Source>
object
downcast_vehicle(const Source &contained)
{
  // Just construct a new container around this.
  return object(Result(contained));
}

// Specialise for not2tc: because of the built-in ambiguity inre whether not2tc
// constructs a new not2t object, or copies one, we actually use not2tc in a
// useful manner here. So engage in type mangling.
template <>
object
downcast_vehicle<not2tc, expr2tc>(const expr2tc &contained)
{
  return object(reinterpret_cast<const not2tc &>(contained));
}

object
downcast_type(const type2tc &type)
{
  if (is_nil_type(type))
    return object();

  assert(type->type_id < type2t::end_type_id);
  object o = (*type_to_downcast)[type->type_id];
  return o(type);
}

object
downcast_expr(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return object();

  assert(expr->expr_id < expr2t::end_expr_id);
  object o = (*expr_to_downcast)[expr->expr_id];
  return o(expr);
}

static void
py_deconstructor()
{
  // Release global reference
  delete type_to_downcast;
  delete expr_to_downcast;
  type_to_downcast = NULL;
  expr_to_downcast = NULL;
}

BOOST_PYTHON_MODULE(esbmc)
{
  // This is essentially the entry point for the esbmc shared object.
  // Workarounds for the static order initialization are in init_esbmc_process
  // due to some annoyance with object lifetime.
  scope esbmc;

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
    object types(handle<>(borrowed(PyImport_AddModule("esbmc.type"))));
    scope quux = types;

    esbmc.attr("type") = types;

    build_base_type2t_python_class();

    types.attr("is_nil_type") = make_function(&is_nil_type);

    // In this scope, define the old irep types to
    class_<typet>("typet", no_init);
    class_<code_typet, bases<typet> >("code_typet", no_init);

#define _ESBMC_IREP2_MPL_TYPE_SET(r, data, elem) BOOST_PP_CAT(elem,_type2t)::build_python_class(type2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(_ESBMC_IREP2_MPL_TYPE_SET, foo, ESBMC_LIST_OF_TYPES)

    build_type2t_container_converters();

    // Build downcasting infrastructure
    type_to_downcast = new dict();
#define _ESBMC_IREP2_TYPE_DOWNCASTING(r, data, elem) \
    (*type_to_downcast)[type2t::BOOST_PP_CAT(elem,_id)] = \
        make_function(downcast_vehicle<BOOST_PP_CAT(elem,_type2tc), type2tc>);
BOOST_PP_LIST_FOR_EACH(_ESBMC_IREP2_TYPE_DOWNCASTING, foo, ESBMC_LIST_OF_TYPES)
  }

  {
    object exprs(handle<>(borrowed(PyImport_AddModule("esbmc.expr"))));
    scope quux = exprs;

    esbmc.attr("expr") = exprs;

    exprs.attr("is_nil_expr") = make_function(&is_nil_expr);

    // Define old expr class too
    class_<exprt>("exprt", no_init);

    build_base_expr2t_python_class();
#define _ESBMC_EXPR2_MPL_EXPR_SET(r, data, elem) BOOST_PP_CAT(elem,2t)::build_python_class(expr2t::BOOST_PP_CAT(elem,_id));
BOOST_PP_LIST_FOR_EACH(_ESBMC_EXPR2_MPL_EXPR_SET, foo, ESBMC_LIST_OF_EXPRS)

    build_expr2t_container_converters();

    // Build downcasting infrastructure
    expr_to_downcast = new dict();
#define _ESBMC_IREP2_EXPR_DOWNCASTING(r, data, elem) \
    (*expr_to_downcast)[expr2t::BOOST_PP_CAT(elem,_id)] = \
        make_function(downcast_vehicle<BOOST_PP_CAT(elem,2tc), expr2tc>);
BOOST_PP_LIST_FOR_EACH(_ESBMC_IREP2_EXPR_DOWNCASTING, foo, ESBMC_LIST_OF_EXPRS)

  }

  // Register BigInt globally
  build_bigint_python_class();
  build_guard_python_class();
  build_dstring_python_class();

  // Alas, we need to pass handles to optionst, namespace, goto funcs around.
  // User should be able to extract them from whatever execution context they
  // generate.
  class_<optionst>("optionst", no_init); // basically opaque
  class_<namespacet>("namespacet", no_init); // basically opaque

  // Build smt solver related stuff
  build_smt_conv_python_class();

  // Build goto function class representions.
  build_goto_func_class();

  // Build fixedbvt class
  build_fixedbv_python_class();

  // Register old-irep to new-irep converters
  esbmc_python_cvt<type2tc, typet, false, true, false, migrate_func<type2tc> >();
  esbmc_python_cvt<expr2tc, exprt, false, true, false, migrate_func<expr2tc> >();

  // And backwards
  esbmc_python_cvt<typet, type2tc, false, true, false, migrate_func<typet> >();
  esbmc_python_cvt<exprt, expr2tc, false, true, false, migrate_func<exprt> >();

  // Locationt objects now...
  class_<locationt>("locationt", no_init);
  init<irep_idt, unsigned int, unsigned int, irep_idt> location2t_init;
  class_<location>("location", location2t_init)
    .def_readwrite("file", &location::file)
    .def_readwrite("line", &location::line)
    .def_readwrite("column", &location::column)
    .def_readwrite("function", &location::function)
    .def("from_locationt", &location::from_locationt)
    .staticmethod("from_locationt");

  build_goto_symex_classes();

  def("downcast_type", &downcast_type);
  def("downcast_expr", &downcast_expr);

  // Ugh.
  class_<contextt, boost::noncopyable>("contextt", no_init); // basically opaque
  class_<message_handlert, boost::noncopyable>("message_handler", no_init); // basically opaque
  class_<ui_message_handlert, boost::noncopyable, bases<message_handlert> >("ui_message_handler", no_init); // basically opaque
  class_<cbmc_parseoptionst, boost::noncopyable>("parseoptions", no_init)
    .def_readwrite("goto_functions", &cbmc_parseoptionst::goto_functions)
    .def_readonly("message_handler", &language_uit::ui_message_handler)
    .def_readonly("context", &language_uit::context);

  build_value_set_classes();

  def("trap", &its_a_trap);

  object atexit = import("atexit");
  atexit.attr("register")(make_function(py_deconstructor));
}

// Include these other things that are special to the esbmc binary:

#if 0
const mode_table_et mode_table[] =
{
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CLANG_C,
#endif
  LANGAPI_HAVE_MODE_C,
#ifndef WITHOUT_CLANG
  LANGAPI_HAVE_MODE_CLANG_CPP,
#endif
  LANGAPI_HAVE_MODE_CPP,
  LANGAPI_HAVE_MODE_END
};

extern "C" uint8_t buildidstring_buf[1];
uint8_t *version_string = buildidstring_buf;
#endif
