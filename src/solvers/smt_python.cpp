#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <functional>
#include <solvers/smt/smt_conv.h>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <solve.h>
#include <smt_python.h>

class dummy_solver_class
{
};
class dummy_solver_class2
{
};
class dummy_solver_class3
{
};

// Template for checking whether the return value from get_override is none
// or not. Placed in class to make it a friend of the wrappers.
class get_override_checked_class
{
public:
  template <typename T>
  static inline boost::python::override
  get_override_checked(const T *x, const char *name)
  {
    using namespace boost::python;
    override f = x->get_override(name);
    if(f.is_none())
    {
      std::cerr << "Pure virtual method \"" << name
                << "\" called during solving"
                   "; you need to override it from python\n";
      abort();
    }

    return f;
  }
};

template <typename T>
inline boost::python::override
get_override_checked(const T *x, const char *name)
{
  return get_override_checked_class::get_override_checked(x, name);
}

// Convert an incoming ast / sort into a handle for a python object, extracted
// out of the wrapper class each inherits from. I was kind of expecting b.p
// to be doing this itself, but apparently not, seeing how it unwraps many
// things before manipulating them.
// This conversion entirely fits with the smt_convt model of doing things:
// asts and sorts merrily get passed around, but only become their derived type
// when the derived smt_convt representing the solver gets a hold of it. It's
// just a coincidence that that derived smt_convt is in a managed environment.
#define ast_down(x) smt_ast_wrapper::cast_ast_down((x))
#define sort_down(x) smt_sort_wrapper::cast_sort_down((x))
#define conv_down(x) smt_convt_wrapper::cast_conv_down((x))

template <typename... Args>
smt_sort_wrapper::smt_sort_wrapper(Args... args) : smt_sort(args...)
{
}

boost::python::object smt_sort_wrapper::cast_sort_down(smt_sortt s)
{
  using namespace boost::python;
  const smt_sort_wrapper *sort = dynamic_cast<const smt_sort_wrapper *>(s);
  assert(
    sort != NULL &&
    "All sorts reaching smt_convt wrapper should be sort wrappers");
  PyObject *obj = boost::python::detail::wrapper_base_::get_owner(*sort);
  assert(obj != NULL && "Wrapped SMT Sort doesn't have a wrapped PyObject?");
  handle<> h(borrowed(obj));
  object o(h);
  return o;
}

smt_ast_wrapper::smt_ast_wrapper(smt_convt *ctx, smt_sortt s) : smt_ast(ctx, s)
{
  assert(dynamic_cast<const smt_sort_wrapper *>(s) != NULL);
}

boost::python::object smt_ast_wrapper::cast_ast_down(smt_astt a)
{
  using namespace boost::python;
  const smt_ast_wrapper *ast = dynamic_cast<const smt_ast_wrapper *>(a);
  assert(
    ast != NULL &&
    "All asts reaching smt_convt wrapper should be ast wrappers");
  PyObject *obj = boost::python::detail::wrapper_base_::get_owner(*ast);
  assert(obj != NULL && "Wrapped SMT AST doesn't have a wrapped PyObject?");
  handle<> h(borrowed(obj));
  object o(h);
  return o;
}

smt_astt
smt_ast_wrapper::ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
{
  using namespace boost::python;
  if(override f = get_override_checked(this, "ite"))
    return f(conv_down(ctx), ast_down(cond), ast_down(falseop));

  return smt_ast::ite(ctx, cond, falseop);
}

smt_astt smt_ast_wrapper::default_ite(
  smt_convt *ctx,
  smt_astt cond,
  smt_astt falseop) const
{
  return smt_ast::ite(ctx, cond, falseop);
}

smt_astt smt_ast_wrapper::eq(smt_convt *ctx, smt_astt other) const
{
  using namespace boost::python;
  if(override f = get_override_checked(this, "eq"))
    return f(conv_down(ctx), ast_down(other));

  return smt_ast::eq(ctx, other);
}

smt_astt smt_ast_wrapper::default_eq(smt_convt *ctx, smt_astt other) const
{
  return smt_ast::eq(ctx, other);
}

void smt_ast_wrapper::assign(smt_convt *ctx, smt_astt sym) const
{
  using namespace boost::python;
  if(override f = this->get_override("assign"))
    f(conv_down(ctx), ast_down(sym));
  else
    smt_ast::assign(ctx, sym);
}

void smt_ast_wrapper::default_assign(smt_convt *ctx, smt_astt sym) const
{
  smt_ast::assign(ctx, sym);
}

smt_astt smt_ast_wrapper::update(
  smt_convt *ctx,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  using namespace boost::python;
  if(override f = get_override_checked(this, "update"))
    return f(conv_down(ctx), ast_down(value), idx, idx_expr);

  return smt_ast::update(ctx, value, idx, idx_expr);
}

smt_astt smt_ast_wrapper::default_update(
  smt_convt *ctx,
  smt_astt value,
  unsigned int idx,
  expr2tc idx_expr) const
{
  return smt_ast::update(ctx, value, idx, idx_expr);
}

smt_astt smt_ast_wrapper::select(smt_convt *ctx, const expr2tc &idx) const
{
  using namespace boost::python;
  if(override f = get_override_checked(this, "select"))
    return f(conv_down(ctx), idx);

  return smt_ast::select(ctx, idx);
}

smt_astt
smt_ast_wrapper::default_select(smt_convt *ctx, const expr2tc &idx) const
{
  return smt_ast::select(ctx, idx);
}

smt_astt smt_ast_wrapper::project(smt_convt *ctx, unsigned int elem) const
{
  using namespace boost::python;
  if(override f = get_override_checked(this, "project"))
    return f(conv_down(ctx), elem);

  return smt_ast::project(ctx, elem);
}

smt_astt
smt_ast_wrapper::default_project(smt_convt *ctx, unsigned int elem) const
{
  return smt_ast::project(ctx, elem);
}

smt_convt_wrapper::smt_convt_wrapper(
  bool int_encoding,
  const namespacet &_ns,
  bool bools_in_arrays,
  bool can_init_inf_arrays)
  : smt_convt(int_encoding, _ns),
    array_iface(bools_in_arrays, can_init_inf_arrays),
    tuple_iface()
{
  // Overriding solver in python needs to implement these ifaces
  set_tuple_iface(this);
  set_array_iface(this);
}

void smt_convt_wrapper::assert_ast(smt_astt a)
{
  get_override_checked(this, "assert_ast")(ast_down(a));
}

smt_convt::resultt smt_convt_wrapper::dec_solve()
{
  return get_override_checked(this, "dec_solve")();
}

const std::string smt_convt_wrapper::solver_text()
{
  return get_override_checked(this, "solver_text")();
}

tvt smt_convt_wrapper::l_get(smt_astt a)
{
  return get_override_checked(this, "l_get")(ast_down(a));
}

smt_astt smt_convt_wrapper::mk_smt_int(const mp_integer &theint, bool sign)
{
  return get_override_checked(this, "mk_smt_int")(theint, sign);
}

smt_astt smt_convt_wrapper::mk_smt_bool(bool val)
{
  return get_override_checked(this, "mk_smt_bool")(val);
}

smt_astt smt_convt_wrapper::mk_smt_symbol(const std::string &name, smt_sortt s)
{
  return get_override_checked(this, "mk_smt_symbol")(name, sort_down(s));
}

smt_astt smt_convt_wrapper::mk_smt_real(const std::string &str)
{
  return get_override_checked(this, "mk_smt_real")(str);
}

smt_astt smt_convt_wrapper::mk_smt_bv(const mp_integer &theint, smt_sortt s)
{
  return get_override_checked(this, "mk_smt_bv")(s, theint);
}

bool smt_convt_wrapper::get_bool(smt_astt a)
{
  return get_override_checked(this, "get_bool")(ast_down(a));
}

BigInt smt_convt_wrapper::get_bv(smt_astt a)
{
  return get_override_checked(this, "get_bv")(ast_down(a));
}

/*************************** Array API ***********************************/
smt_astt smt_convt_wrapper::mk_array_symbol(
  const std::string &name,
  smt_sortt sort,
  smt_sortt subtype)
{
  return get_override_checked(this, "mk_array_symbol")(
    name, sort_down(sort), sort_down(subtype));
}

expr2tc smt_convt_wrapper::get_array_elem(
  smt_astt a,
  uint64_t idx,
  const type2tc &subtype)
{
  return get_override_checked(this, "get_array_elem")(
    ast_down(a), idx, subtype);
}

const smt_ast *smt_convt_wrapper::convert_array_of(
  smt_astt init_val,
  unsigned long domain_width)
{
  // XXX a default is provided by array_iface.
  using namespace boost::python;
  if(override f = this->get_override("convert_array_of"))
    return f(ast_down(init_val), domain_width);

  return default_convert_array_of(init_val, domain_width, this);
}

void smt_convt_wrapper::add_array_constraints_for_solving()
{
  get_override_checked(this, "add_array_constraints_for_solving")();
}

void smt_convt_wrapper::push_array_ctx(void)
{
  std::cerr
    << "Push/pop using python-extended solver isn't supported right now\n";
  abort();
}

void smt_convt_wrapper::pop_array_ctx(void)
{
  std::cerr
    << "Push/pop using python-extended solver isn't supported right now\n";
  abort();
}

/*************************** Tuple API ***********************************/
smt_sortt smt_convt_wrapper::mk_struct_sort(const type2tc &type)
{
  return get_override_checked(this, "mk_struct_sort")(type);
}

smt_astt smt_convt_wrapper::tuple_create(const expr2tc &structdef)
{
  return get_override_checked(this, "tuple_create")(structdef);
}

smt_astt smt_convt_wrapper::tuple_fresh(smt_sortt s, std::string name)
{
  return get_override_checked(this, "tuple_fresh")(sort_down(s), name);
}

smt_astt smt_convt_wrapper::tuple_array_create(
  const type2tc &array_type,
  smt_astt *inputargs,
  bool const_array,
  smt_sortt domain)
{
  using namespace boost::python;
  const array_type2t &arr_ref = to_array_type(array_type);
  assert(arr_ref.subtype->type_id == type2t::struct_id);

  const struct_type2t &struct_ref = to_struct_type(arr_ref.subtype);

  list l;
  if(const_array)
  {
    // There's only one ast.
    l.append(ast_down(inputargs[0]));
  }
  else
  {
    for(unsigned int i = 0; i < struct_ref.members.size(); i++)
      l.append(ast_down(inputargs[i]));
  }

  return tuple_array_create_remangled(array_type, l, const_array, domain);
}

smt_astt smt_convt_wrapper::tuple_array_create_remangled(
  const type2tc &array_type,
  boost::python::object l,
  bool const_array,
  smt_sortt domain)
{
  return get_override_checked(this, "tuple_array_create")(
    array_type, l, const_array, sort_down(domain));
}

smt_astt smt_convt_wrapper::tuple_array_of(
  const expr2tc &init_value,
  unsigned long domain_width)
{
  return get_override_checked(this, "tuple_array_of")(init_value, domain_width);
}

smt_astt
smt_convt_wrapper::mk_tuple_symbol(const std::string &name, smt_sortt s)
{
  return get_override_checked(this, "mk_tuple_symbol")(name, sort_down(s));
}

smt_astt smt_convt_wrapper::mk_tuple_array_symbol(const expr2tc &expr)
{
  return get_override_checked(this, "mk_tuple_array_symbol")(expr);
}

expr2tc smt_convt_wrapper::tuple_get(const expr2tc &expr)
{
  return get_override_checked(this, "tuple_get")(expr);
}

void smt_convt_wrapper::add_tuple_constraints_for_solving()
{
  get_override_checked(this, "add_tuple_constraints_for_solving")();
}

void smt_convt_wrapper::push_tuple_ctx()
{
  std::cerr
    << "Push/pop using python-extended solver isn't supported right now\n";
  abort();
}

void smt_convt_wrapper::pop_tuple_ctx()
{
  std::cerr
    << "Push/pop using python-extended solver isn't supported right now\n";
  abort();
}

// Method for casting an smt_convt down to the wrapped type.
boost::python::object smt_convt_wrapper::cast_conv_down(smt_convt *c)
{
  using namespace boost::python;
  smt_convt_wrapper *conv = dynamic_cast<smt_convt_wrapper *>(c);
  assert(conv != NULL && "smt_convt handed to ast is not python wrapped?");
  PyObject *obj = boost::python::detail::wrapper_base_::get_owner(*conv);
  assert(obj != NULL && "Wrapped SMT convt doesn't have a wrapped PyObject?");
  handle<> h(borrowed(obj));
  object o(h);
  return o;
}

static smt_convt *bounce_solver_factory(
  bool int_encoding,
  const namespacet &ns,
  const optionst &options,
  const char *name = "bees")
{
  std::string foo(name);
  return create_solver_factory(name, int_encoding, ns, options);
}

static smt_ast *bounce_convert_ast(smt_convt *conv, const expr2tc &expr)
{
  // It's funny because b.p tends to not work with anything 'const' :/.
  // A certain amount of protection is lost byh this; however so long as b.p
  // needs access to the return of convert_ast, that's always going to be lost.
  return const_cast<smt_ast *>(conv->convert_ast(expr));
}

class const_smt_sort_to_python
{
public:
  static PyObject *convert(smt_sortt s)
  {
    using namespace boost::python;
    smt_sort *ss = const_cast<smt_sort *>(s);
    return incref(object(ss).ptr());
  }
};

class const_smt_ast_to_python
{
public:
  static PyObject *convert(smt_astt a)
  {
    using namespace boost::python;
    smt_ast *aa = const_cast<smt_ast *>(a);
    return incref(object(aa).ptr());
  }
};

boost::python::object downcast_sort(smt_sort *s)
{
  return sort_down(s);
}

boost::python::object downcast_ast(smt_ast *a)
{
  return ast_down(a);
}

// After all your wickedness boost.python, woe, woe unto you. If b.p is told
// simply that the sort field of smt_ast's is a smt_sort, then it won't perform
// any kind of unwrapping step to find the subclass. That's not too bad, but
// then if we _manually_ downcast it from python, some kind of caching
// mechanism (which rightly assumes a PyObject has only one type) prevents us
// from recognizing the subclass, and downcasting fails.
// Fix this by creating an accessor that unwraps the PyObject.
boost::python::object get_sort_from_ast(smt_ast *a)
{
  return sort_down(a->sort);
}

void build_smt_conv_python_class(void)
{
  using namespace boost::python;

  scope esbmc;

  object solve(handle<>(borrowed(PyImport_AddModule("esbmc.solve"))));
  scope quux = solve;

  esbmc.attr("solve") = solve;

  solve.attr("downcast_sort") = make_function(&downcast_sort);
  solve.attr("downcast_ast") = make_function(&downcast_ast);

  // It would appear people _do_ want to write sovlers from python.

  enum_<tvt::tv_enumt>("tvt_enum")
    .value("true", tvt::tv_enumt::TV_TRUE)
    .value("false", tvt::tv_enumt::TV_FALSE)
    .value("unknown", tvt::tv_enumt::TV_UNKNOWN);

  class_<tvt>("tvt")
    .def(init<bool>())
    .def(init<tvt::tv_enumt>())
    .def("is_true", &tvt::is_true)
    .def("is_false", &tvt::is_false)
    .def("is_unknown", &tvt::is_unknown)
    .def("is_known", &tvt::is_known)
    .def("invert", &tvt::invert);

  enum_<smt_sort_kind>("smt_sort_kind")
    .value("int", SMT_SORT_INT)
    .value("real", SMT_SORT_REAL)
    .value("sbv", SMT_SORT_BV)
    .value("fixedbv", SMT_SORT_FIXEDBV)
    .value("array", SMT_SORT_ARRAY)
    .value("bool", SMT_SORT_BOOL)
    .value("struct", SMT_SORT_STRUCT);

  class_<smt_sort_wrapper, boost::noncopyable>(
    "smt_sort", init<smt_sort_kind>())
    .def(init<smt_sort_kind, unsigned long>())
    .def(init<smt_sort_kind, unsigned long, unsigned long>())
    .def_readwrite("id", &smt_sort::id)
    .def("get_data_width", &smt_sort::get_data_width)
    .def("get_domain_width", &smt_sort::get_domain_width)
    .def("get_significand_width", &smt_sort::get_significand_width);

  // Declare smt_ast class, wrapped, with overrides available. Note that these
  // are all declared to return internal references: by default is's the solvers
  // problem to memory manage. If overridden, then it's the overriders problem
  // to keep a python reference to all smt_ast's that C++ might point at.
  typedef return_internal_reference<> rte;
  class_<smt_ast_wrapper, boost::noncopyable>(
    "smt_ast", init<smt_convt *, smt_sortt>())
    .add_property("sort", make_function(&get_sort_from_ast))
    .def("ite", &smt_ast_wrapper::ite, &smt_ast_wrapper::default_ite, rte())
    .def("eq", &smt_ast_wrapper::eq, &smt_ast_wrapper::default_eq, rte())
    .def("assign", &smt_ast_wrapper::assign, &smt_ast_wrapper::default_assign)
    .def(
      "update",
      &smt_ast_wrapper::update,
      &smt_ast_wrapper::default_update,
      rte())
    .def(
      "select",
      &smt_ast_wrapper::select,
      &smt_ast_wrapper::default_select,
      rte())
    .def(
      "project",
      &smt_ast_wrapper::project,
      &smt_ast_wrapper::default_project,
      rte());

  smt_astt (smt_convt::*mk_smt_bv)(const mp_integer &, std::size_t) =
    &smt_convt::mk_smt_bv;

  // Register generic smt_convt facilities: only allow the python user to do
  // expression conversion. Any new smt_convt implementation should be done
  // in C++ for example.
  // Refrain from registering enums too: basic implementation pls.
  class_<smt_convt_wrapper, boost::noncopyable>(
    "smt_convt", init<bool, const namespacet &, bool, bool>())
    .def_readonly("pointer_struct", &smt_convt::pointer_struct)
    .def("smt_post_init", &smt_convt::smt_post_init)
    .def("convert_sort", &smt_convt::convert_sort, rte())
    .def("convert_ast", &bounce_convert_ast, rte())
    .def("push_ctx", &smt_convt::push_ctx)
    .def("pop_ctx", &smt_convt::pop_ctx)
    .def("convert_assign", &smt_convt::convert_assign, rte())
    .def("invert_ast", &smt_convt::invert_ast, rte())
    .def("imply_ast", &smt_convt::imply_ast, rte())
    .def("assert_ast", &smt_convt::assert_ast)
    .def("dec_solve", &smt_convt::dec_solve)
    .def("get", &smt_convt::get)
    .def(
      "calculate_array_domain_width", &smt_convt::calculate_array_domain_width)
    .def("assert_ast", pure_virtual(&smt_convt::assert_ast))
    .def("dec_solve", pure_virtual(&smt_convt::dec_solve))
    .def("l_get", pure_virtual(&smt_convt::l_get))
    .def("mk_smt_int", pure_virtual(&smt_convt::mk_smt_int), rte())
    .def("mk_smt_bool", pure_virtual(&smt_convt::mk_smt_bool), rte())
    .def("mk_smt_symbol", pure_virtual(&smt_convt::mk_smt_symbol), rte())
    .def("mk_smt_real", pure_virtual(&smt_convt::mk_smt_real), rte())
    .def("get_bool", pure_virtual(&smt_convt::get_bool))
    .def("get_bv", pure_virtual(&smt_convt::get_bv))
    .def("mk_extract", pure_virtual(&smt_convt::mk_extract), rte())
    .def(
      "tuple_array_create",
      pure_virtual(&smt_convt_wrapper::tuple_array_create_remangled),
      rte())
    .def("mk_smt_bv", mk_smt_bv, rte());

  // Result enum for solving
  enum_<smt_convt::resultt>("smt_result")
    .value("sat", smt_convt::resultt::P_SATISFIABLE)
    .value("unsat", smt_convt::resultt::P_UNSATISFIABLE)
    .value("error", smt_convt::resultt::P_ERROR)
    .value("smtlib", smt_convt::resultt::P_SMTLIB);

  // ast_vec registration, as a vector
  class_<smt_convt::ast_vec>("smt_ast_vec")
    .def(vector_indexing_suite<smt_convt::ast_vec>());

  // Now register factories for the set of solvers that have been built in.
  // These are going to return raw smt_convt pointers, and we can't have boost.
  // python store them by value without defining (and copy constructing) all
  // converter classes. Thus we're left to the manage_new_object return policy,
  // which means when the python object expires the solver is deleted. As a
  // result, it probably shouldn't be embedded into any C++ objects except
  // with great care.
  scope solve2 = class_<dummy_solver_class2>("solvers");
  type_info dummy_class_id = type_id<dummy_solver_class3>();
  for(unsigned int i = 0; i < esbmc_num_solvers; i++)
  {
    std::stringstream ss;
    const std::string &solver_name = esbmc_solvers[i].name;

    // Here, we iteratively declare classes to boost.python as we discover
    // new C++ solver classes. But there's a problem -- boost is aware of the
    // runtime type_id of each class that's registered, notices that we're
    // registering the same thing more than once, and complains on stderr.
    // To get around this, call the base object constructor that registers the
    // new class (class_base), but not the higher level class_ object that
    // tries to register converters and causes the errors on stderr.
    // Or as I like to call it, "duckrolling in a duckroll thread without being
    // duckrolled".
    auto solve = objects::class_base(solver_name.c_str(), 1, &dummy_class_id);
    scope _solve = solve; // This should (?) ensure the def below gets scoped?

    // Interpret our class_base as a class_. The only actual difference is
    // what constructors get called, there's no change in storage.
    // This is liable to break horribly with future versions of boost.python.
    class_<dummy_solver_class3> *cp =
      reinterpret_cast<class_<dummy_solver_class3> *>(&solve);

    // Trolpocolypse: we don't have a static function to create each solver,
    // or at least not one that doesn't involve mangling tuple_apis and the
    // like. So: use one function and tell python it has a default argument.
    // Use that to work out what solver to create. A possible alternative would
    // be to store a python object holding the solver name, and define a method
    // to construct it or something.
    cp->def(
      "make",
      &bounce_solver_factory,
      (arg("int_encoding"),
       arg("ns"),
       arg("options"),
       arg("name") = solver_name),
      return_value_policy<manage_new_object>());

    cp->staticmethod("make");
  }

  // Set right up step right up to see the boost.python conversion circus, in
  // town today only. Our const-correctness model means that we need to have
  // various smt_sort * fields const-qualified. But boost.python won't convert
  // that into a python object because... it's const qualified! Which is safe,
  // but irritating. Therefore we need to write/use a converter for converting
  // const smt_sort*'s to smt_sort* objects.
  to_python_converter<smt_sortt, const_smt_sort_to_python>();
  to_python_converter<smt_astt, const_smt_ast_to_python>();
}

smt_astt smt_convt_wrapper::mk_smt_fpbv(const ieee_floatt &thereal)
{
  return get_override_checked(this, "mk_smt_fpbv")(thereal);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  return get_override_checked(this, "mk_smt_fpbv_nan")(ew, sw);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  return get_override_checked(this, "mk_smt_fpbv_inf")(sgn, ew, sw);
}

smt_sortt smt_convt_wrapper::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  return get_override_checked(this, "mk_fpbv_sort")(ew, sw);
}

smt_sortt smt_convt_wrapper::mk_fpbv_rm_sort()
{
  return get_override_checked(this, "mk_fpbv_rm_sort")();
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  return get_override_checked(this, "mk_smt_fpbv_rm")(rm);
}

smt_astt smt_convt_wrapper::mk_add(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_add")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvadd(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvadd")(a, b);
}

smt_astt smt_convt_wrapper::mk_sub(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_sub")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvsub(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvsub")(a, b);
}

smt_astt smt_convt_wrapper::mk_mul(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_mul")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvmul(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvmul")(a, b);
}

smt_astt smt_convt_wrapper::mk_mod(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_mod")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvsmod(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvsmod")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvumod(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvumod")(a, b);
}

smt_astt smt_convt_wrapper::mk_div(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_div")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvsdiv(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvsdiv")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvudiv(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvudiv")(a, b);
}

smt_astt smt_convt_wrapper::mk_shl(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_shl")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvshl(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvshl")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvashr(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvashr")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvlshr(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvlshr")(a, b);
}

smt_astt smt_convt_wrapper::mk_neg(smt_astt a)
{
  return get_override_checked(this, "mk_neg")(a);
}

smt_astt smt_convt_wrapper::mk_bvneg(smt_astt a)
{
  return get_override_checked(this, "mk_bvneg")(a);
}

smt_astt smt_convt_wrapper::mk_bvnot(smt_astt a)
{
  return get_override_checked(this, "mk_bvnot")(a);
}

smt_astt smt_convt_wrapper::mk_bvnxor(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvnxor")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvnor(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvnor")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvnand(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvnand")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvxor(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvxor")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvor(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvor")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvand(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvand")(a, b);
}

smt_astt smt_convt_wrapper::mk_implies(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_implies")(a, b);
}

smt_astt smt_convt_wrapper::mk_xor(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_xor")(a, b);
}

smt_astt smt_convt_wrapper::mk_or(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_or")(a, b);
}

smt_astt smt_convt_wrapper::mk_and(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_and")(a, b);
}

smt_astt smt_convt_wrapper::mk_not(smt_astt a)
{
  return get_override_checked(this, "mk_not")(a);
}

smt_astt smt_convt_wrapper::mk_lt(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_lt")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvult(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvult")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvslt(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvslt")(a, b);
}

smt_astt smt_convt_wrapper::mk_gt(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_gt")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvugt(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvugt")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvsgt(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvsgt")(a, b);
}

smt_astt smt_convt_wrapper::mk_le(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_le")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvule(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvule")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvsle(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvsle")(a, b);
}

smt_astt smt_convt_wrapper::mk_ge(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_ge")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvuge(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvuge")(a, b);
}

smt_astt smt_convt_wrapper::mk_bvsge(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_bvsge")(a, b);
}

smt_astt smt_convt_wrapper::mk_eq(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_eq")(a, b);
}

smt_astt smt_convt_wrapper::mk_neq(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_neq")(a, b);
}

smt_astt smt_convt_wrapper::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  return get_override_checked(this, "mk_store")(a, b, c);
}

smt_astt smt_convt_wrapper::mk_select(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_select")(a, b);
}

smt_astt smt_convt_wrapper::mk_real2int(smt_astt a)
{
  return get_override_checked(this, "mk_real2int")(a);
}

smt_astt smt_convt_wrapper::mk_int2real(smt_astt a)
{
  return get_override_checked(this, "mk_int2real")(a);
}

smt_astt smt_convt_wrapper::mk_isint(smt_astt a)
{
  return get_override_checked(this, "mk_isint")(a);
}

smt_astt smt_convt_wrapper::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  return get_override_checked(this, "mk_sign_ext")(a, topwidth);
}

smt_astt smt_convt_wrapper::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  return get_override_checked(this, "mk_zero_ext")(a, topwidth);
}

smt_astt smt_convt_wrapper::mk_smt_typecast_from_fpbv_to_ubv(
  smt_astt from,
  std::size_t width)
{
  return get_override_checked(this, "mk_smt_typecast_from_fpbv_to_ubv")(
    from, width);
}

smt_astt smt_convt_wrapper::mk_smt_typecast_from_fpbv_to_sbv(
  smt_astt from,
  std::size_t width)
{
  return get_override_checked(this, "mk_smt_typecast_from_fpbv_to_sbv")(
    from, width);
}

smt_astt smt_convt_wrapper::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return get_override_checked(this, "mk_smt_typecast_from_fpbv_to_sbv")(
    from, to, rm);
}

smt_astt smt_convt_wrapper::mk_smt_typecast_ubv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return get_override_checked(this, "mk_smt_typecast_ubv_to_fpbv")(
    from, to, rm);
}

smt_astt smt_convt_wrapper::mk_smt_typecast_sbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  return get_override_checked(this, "mk_smt_typecast_sbv_to_fpbv")(
    from, to, rm);
}

smt_astt
smt_convt_wrapper::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  return get_override_checked(this, "mk_smt_nearbyint_from_float")(from, rm);
}

smt_astt
smt_convt_wrapper::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return get_override_checked(this, "mk_smt_fpbv_add")(lhs, rhs, rm);
}

smt_astt
smt_convt_wrapper::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return get_override_checked(this, "mk_smt_fpbv_sub")(lhs, rhs, rm);
}

smt_astt
smt_convt_wrapper::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return get_override_checked(this, "mk_smt_fpbv_mul")(lhs, rhs, rm);
}

smt_astt
smt_convt_wrapper::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  return get_override_checked(this, "mk_smt_fpbv_div")(lhs, rhs, rm);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  return get_override_checked(this, "mk_smt_fpbv_sqrt")(rd, rm);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_fma(
  smt_astt v1,
  smt_astt v2,
  smt_astt v3,
  smt_astt rm)
{
  return get_override_checked(this, "mk_smt_fpbv_fma")(v1, v2, v3, rm);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  return get_override_checked(this, "mk_smt_fpbv_eq")(lhs, rhs);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_gt(smt_astt lhs, smt_astt rhs)
{
  return get_override_checked(this, "mk_smt_fpbv_gt")(lhs, rhs);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_lt(smt_astt lhs, smt_astt rhs)
{
  return get_override_checked(this, "mk_smt_fpbv_lt")(lhs, rhs);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_gte(smt_astt lhs, smt_astt rhs)
{
  return get_override_checked(this, "mk_smt_fpbv_gte")(lhs, rhs);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_lte(smt_astt lhs, smt_astt rhs)
{
  return get_override_checked(this, "mk_smt_fpbv_lte")(lhs, rhs);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_is_nan(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_is_nan")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_is_inf(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_is_inf")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_is_normal(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_is_normal")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_is_zero(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_is_zero")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_is_negative(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_is_negative")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_is_positive(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_is_positive")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_abs(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_abs")(op);
}

smt_astt smt_convt_wrapper::mk_smt_fpbv_neg(smt_astt op)
{
  return get_override_checked(this, "mk_smt_fpbv_neg")(op);
}

ieee_floatt smt_convt_wrapper::get_fpbv(smt_astt a)
{
  return get_override_checked(this, "get_fpbv")(a);
}

smt_astt smt_convt_wrapper::mk_from_bv_to_fp(smt_astt op, smt_sortt to)
{
  return get_override_checked(this, "mk_from_bv_to_fp")(op, to);
}

smt_astt smt_convt_wrapper::mk_from_fp_to_bv(smt_astt op)
{
  return get_override_checked(this, "mk_from_fp_to_bv")(op);
}

smt_astt
smt_convt_wrapper::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  return get_override_checked(this, "mk_extract")(a, high, low);
}

smt_astt smt_convt_wrapper::mk_concat(smt_astt a, smt_astt b)
{
  return get_override_checked(this, "mk_concat")(a, b);
}

smt_astt smt_convt_wrapper::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  return get_override_checked(this, "mk_ite")(cond, t, f);
}

smt_sortt smt_convt_wrapper::mk_bool_sort()
{
  return get_override_checked(this, "mk_bool_sort")();
}

smt_sortt smt_convt_wrapper::mk_real_sort()
{
  return get_override_checked(this, "mk_real_sort")();
}

smt_sortt smt_convt_wrapper::mk_int_sort()
{
  return get_override_checked(this, "mk_int_sort")();
}

smt_sortt smt_convt_wrapper::mk_bv_sort(std::size_t width)
{
  return get_override_checked(this, "mk_bv_sort")(width);
}

smt_sortt smt_convt_wrapper::mk_fbv_sort(std::size_t width)
{
  return get_override_checked(this, "mk_fbv_sort")(width);
}

smt_sortt smt_convt_wrapper::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return get_override_checked(this, "mk_bvfp_sort")(ew, sw);
}

smt_sortt smt_convt_wrapper::mk_bvfp_rm_sort()
{
  return get_override_checked(this, "mk_bvfp_rm_sort")();
}

smt_sortt smt_convt_wrapper::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  return get_override_checked(this, "mk_array_sort")(domain, range);
}

#endif
