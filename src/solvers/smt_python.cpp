#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <functional>

#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_tuple.h>
#include <solvers/smt/smt_array.h>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <util/bp_opaque_ptr.h>

#include "solve.h"

class dummy_solver_class { };
class dummy_solver_class2 { };
class dummy_solver_class3 { };

// Template for checking whether the return value from get_override is none
// or not. Placed in class to make it a friend of the wrappers.
class get_override_checked_class {
public:

template <typename T>
static inline boost::python::override
get_override_checked(const T *x, const char *name)
{
  using namespace boost::python;
  override f = x->get_override(name);
  if (f.is_none()) {
    std::cerr << "Pure virtual method \"" << name << "\" called during solving"
      "; you need to override it from python" << std::endl;
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
#define conv_down(x) smt_convt_wrapper_cvt::cast_conv_down((x))

// Dependency misery: need a definition of a class to extract a PyObject from
// it's wrapper, but smt_ast and smt_convt are mutually dependent. Thus we
// have to make a declaration and put a definition further down the file.
class smt_convt;
class smt_convt_wrapper_cvt
{
public:
  static boost::python::object cast_conv_down(smt_convt *conv);
};

class smt_sort_wrapper : public smt_sort, public boost::python::wrapper<smt_sort>
{
public:
  friend class get_override_checked_class;
  template <typename ...Args>
  smt_sort_wrapper(Args ...args) : smt_sort(args...) { }

  static
  inline boost::python::object
  cast_sort_down(smt_sortt s)
  {
    using namespace boost::python;
    const smt_sort_wrapper *sort = dynamic_cast<const smt_sort_wrapper *>(s);
    assert(sort != NULL && "All sorts reaching smt_convt wrapper should be sort wrappers");
    PyObject *obj = boost::python::detail::wrapper_base_::get_owner(*sort);
    assert(obj != NULL && "Wrapped SMT Sort doesn't have a wrapped PyObject?");
    handle<> h(borrowed(obj));
    object o(h);
    return o;
  }

  virtual ~smt_sort_wrapper() {}
};

class smt_ast_wrapper : public smt_ast, public boost::python::wrapper<smt_ast>
{
public:
  friend class get_override_checked_class;
  template <typename ...Args>
  smt_ast_wrapper(Args ...args) : smt_ast(args...) { }

  static
  inline boost::python::object
  cast_ast_down(smt_astt a)
  {
    using namespace boost::python;
    const smt_ast_wrapper *ast = dynamic_cast<const smt_ast_wrapper *>(a);
    assert(ast != NULL && "All asts reaching smt_convt wrapper should be ast wrappers");
    PyObject *obj = boost::python::detail::wrapper_base_::get_owner(*ast);
    assert(obj != NULL && "Wrapped SMT AST doesn't have a wrapped PyObject?");
    handle<> h(borrowed(obj));
    object o(h);
    return o;
  }

  smt_astt
  ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
  {
    using namespace boost::python;
    if (override f = get_override_checked(this, "ite"))
      return f(ctx, ast_down(cond), ast_down(falseop));
    else
      return smt_ast::ite(ctx, cond, falseop);
  }

  smt_astt default_ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
  {
    return smt_ast::ite(ctx, cond, falseop);
  }

  smt_astt
  eq(smt_convt *ctx, smt_astt other) const
  {
    using namespace boost::python;
    if (override f = get_override_checked(this, "eq"))
      return f(conv_down(ctx), ast_down(other));
    else
      return smt_ast::eq(ctx, other);
  }

  smt_astt
  default_eq(smt_convt *ctx, smt_astt other) const
  {
    return smt_ast::eq(ctx, other);
  }

  void
  assign(smt_convt *ctx, smt_astt sym) const
  {
    using namespace boost::python;
    if (override f = get_override_checked(this, "assign"))
      f(conv_down(ctx), ast_down(sym));
    else
      smt_ast::assign(ctx, sym);
  }

  void
  default_assign(smt_convt *ctx, smt_astt sym) const
  {
    smt_ast::assign(ctx, sym);
  }

  smt_astt
  update(smt_convt *ctx, smt_astt value, unsigned int idx, expr2tc idx_expr = expr2tc()) const
  {
    using namespace boost::python;
    if (override f = get_override_checked(this, "update"))
      return f(conv_down(ctx), ast_down(value), idx, idx_expr);
    else
      return smt_ast::update(ctx, value, idx, idx_expr);
  }

  smt_astt
  default_update(smt_convt *ctx, smt_astt value, unsigned int idx, expr2tc idx_expr = expr2tc()) const
  {
    return smt_ast::update(ctx, value, idx, idx_expr);
  }

  smt_astt
  select(smt_convt *ctx, const expr2tc &idx) const
  {
    using namespace boost::python;
    if (override f = get_override_checked(this, "select"))
      return f(conv_down(ctx), idx);
    else
      return smt_ast::select(ctx, idx);
  }

  smt_astt
  default_select(smt_convt *ctx, const expr2tc &idx) const
  {
    return smt_ast::select(ctx, idx);
  }

  smt_astt
  project(smt_convt *ctx, unsigned int elem) const
  {
    using namespace boost::python;
    if (override f = get_override_checked(this, "project"))
      return f(conv_down(ctx), elem);
    else
      return smt_ast::project(ctx, elem);
  }

  smt_astt
  default_project(smt_convt *ctx, unsigned int elem) const
  {
    return smt_ast::project(ctx, elem);
  }
};

class smt_convt_wrapper : public smt_convt, public array_iface, public tuple_iface, public boost::python::wrapper<smt_convt>
{
public:
  friend class get_override_checked_class;
  friend class smt_convt_wrapper_cvt;
  smt_convt_wrapper(bool int_encoding, const namespacet &_ns, bool is_cpp, bool bools_in_arrays, bool can_init_inf_arrays)
    : smt_convt(int_encoding, _ns, is_cpp),
       array_iface(bools_in_arrays, can_init_inf_arrays),
       tuple_iface()
  {
    // Overriding solver in python needs to implement these ifaces
    set_tuple_iface(this);
    set_array_iface(this);
  }

  smt_astt
  mk_func_app(smt_sortt s, smt_func_kind k, smt_astt const *args, unsigned int numargs)
  {
    // Python is not going to enjoy variable length argument array in any way
    using namespace boost::python;
    list l;
    for (unsigned int i = 0 ;i < numargs; i++)
      l.append(ast_down(args[i]));

    return mk_func_app_remangled(s, k, l);
  }

  smt_astt
  mk_func_app_remangled(smt_sortt s, smt_func_kind k, boost::python::object o)
  {
    return get_override_checked(this, "mk_func_app")(sort_down(s), k, o);
  }

  void
  assert_ast(smt_astt a)
  {
    get_override_checked(this, "assert_ast")(ast_down(a));
  }

  resultt
  dec_solve()
  {
    return get_override_checked(this, "dec_solve")();
  }

  const std::string
  solver_text()
  {
    return get_override_checked(this, "solver_text")();
  }

  tvt
  l_get(smt_astt a)
  {
    return get_override_checked(this, "l_get")(ast_down(a));
  }

  smt_sortt
  mk_sort(const smt_sort_kind k, ...)
  {
    using namespace boost::python;
    // Because this function is variadic (haha poor design choices) we can't
    // just funnel it to python. Extract actual args, then call an overrider.
    va_list ap;
    unsigned long uint;

    boost::python::object o;

    va_start(ap, k);
    switch (k) {
    case SMT_SORT_INT:
    case SMT_SORT_REAL:
    case SMT_SORT_BOOL:
      o = make_tuple(object(k));
      break;
    case SMT_SORT_BV:
      uint = va_arg(ap, unsigned long);
      o = make_tuple(object(k), object(uint));
      break;
    case SMT_SORT_ARRAY:
    {
      smt_sort *dom = va_arg(ap, smt_sort *); // Consider constness?
      smt_sort *range = va_arg(ap, smt_sort *);
      assert(int_encoding || dom->data_width != 0);

      // XXX: setting data_width to 1 if non-bv type?
      // XXX: how are those types going to be convertged to python references eh
      o = make_tuple(object(k), sort_down(dom), sort_down(range));
      break;
    }
    default:
      std::cerr << "Unexpected sort kind " << k << " in smt_convt_wrapper mk_sort" << std::endl;
      abort();
    }

    return mk_sort_remangled(o);
  }

  smt_sortt
  mk_sort_remangled(boost::python::object o)
  {
    return get_override_checked(this, "mk_sort")(o);
  }

  smt_astt
  mk_smt_int(const mp_integer &theint, bool sign)
  {
    return get_override_checked(this, "mk_smt_int")(theint, sign);
  }

  smt_astt
  mk_smt_bool(bool val)
  {
    return get_override_checked(this, "mk_smt_bool")(val);
  }

  smt_astt
  mk_smt_symbol(const std::string &name, smt_sortt s)
  {
    return get_override_checked(this, "mk_smt_symbol")(name, sort_down(s));
  }

  smt_astt
  mk_smt_real(const std::string &str)
  {
    return get_override_checked(this, "mk_smt_real")(str);
  }

  smt_astt
  mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w)
  {
    return get_override_checked(this, "mk_smt_bvint")(theint, sign, w);
  }

  expr2tc
  get_bool(smt_astt a)
  {
    return get_override_checked(this, "get_bool")(ast_down(a));
  }

  expr2tc
  get_bv(const type2tc &t, smt_astt a)
  {
    return get_override_checked(this, "get_bv")(t, ast_down(a));
  }

  smt_astt
  mk_extract(smt_astt a, unsigned int high, unsigned int low, smt_sortt s)
  {
    return get_override_checked(this, "mk_extract")(ast_down(a), high, low, sort_down(s));
  }

  /*************************** Array API ***********************************/
  smt_astt
  mk_array_symbol(const std::string &name, smt_sortt sort, smt_sortt subtype)
  {
    return get_override_checked(this, "mk_array_symbol")(name, sort_down(sort), sort_down(subtype));
  }

  expr2tc
  get_array_elem(smt_astt a, uint64_t idx, const type2tc &subtype)
  {
    return get_override_checked(this, "get_array_elem")(ast_down(a), idx, subtype);
  }

  const smt_ast *
  convert_array_of(smt_astt init_val, unsigned long domain_width)
  {
    // XXX a default is provided by array_iface.
    using namespace boost::python;
    if (override f = this->get_override("convert_array_of"))
      return f(ast_down(init_val), domain_width);
    else
      return default_convert_array_of(init_val, domain_width, this);
  }

  void
  add_array_constraints_for_solving()
  {
    get_override_checked(this, "add_array_constraints_for_solving")();
  }

  void
  push_array_ctx(void)
  {
    std::cerr << "Push/pop using python-extended solver isn't supported right now" << std::endl;
    abort();
  }

  void
  pop_array_ctx(void)
  {
    std::cerr << "Push/pop using python-extended solver isn't supported right now" << std::endl;
    abort();
  }

  /*************************** Tuple API ***********************************/
  smt_sortt
  mk_struct_sort(const type2tc &type)
  {
    return get_override_checked(this, "mk_struct_sort")(type);
  }

  smt_astt
  tuple_create(const expr2tc &structdef)
  {
    return get_override_checked(this, "tuple_create")(structdef);
  }

  smt_astt
  tuple_fresh(smt_sortt s, std::string name = "")
  {
    return get_override_checked(this, "tuple_fresh")(sort_down(s), name);
  }

  smt_astt
  tuple_array_create(const type2tc &array_type, smt_astt *inputargs, bool const_array, smt_sortt domain)
  {
    // XXX XXX XXX this needs to be remangled, array ptr
    return get_override_checked(this, "tuple_array_creaet")(array_type, inputargs, const_array, domain);
  }

  smt_astt
  tuple_array_of(const expr2tc &init_value, unsigned long domain_width)
  {
    return get_override_checked(this, "tuple_array_of")(init_value, domain_width);
  }

  smt_astt
  mk_tuple_symbol(const std::string &name, smt_sortt s)
  {
    return get_override_checked(this, "mk_tuple_symbol")(name, sort_down(s));
  }

  smt_astt
  mk_tuple_array_symbol(const expr2tc &expr)
  {
    return get_override_checked(this, "mk_tuple_array_symbol")(expr);
  }

  expr2tc
  tuple_get(const expr2tc &expr)
  {
    return get_override_checked(this, "tuple_get")(expr);
  }

  void
  add_tuple_constraints_for_solving()
  {
    get_override_checked(this, "add_tuple_constraints_for_solving")();
  }

  void
  push_tuple_ctx()
  {
    std::cerr << "Push/pop using python-extended solver isn't supported right now" << std::endl;
    abort();
  }

  void
  pop_tuple_ctx()
  {
    std::cerr << "Push/pop using python-extended solver isn't supported right now" << std::endl;
    abort();
  }
};

// Method for casting an smt_convt down to the wrapped type.
boost::python::object
smt_convt_wrapper_cvt::cast_conv_down(smt_convt *c)
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

static smt_convt *
bounce_solver_factory(bool is_cpp, bool int_encoding, const namespacet &ns,
    const optionst &options, const char *name = "bees")
{
  std::string foo(name);
  return create_solver_factory(name, is_cpp, int_encoding, ns, options);
}

static
smt_ast *
bounce_convert_ast(smt_convt *conv, const expr2tc &expr)
{
  // It's funny because b.p tends to not work with anything 'const' :/.
  // A certain amount of protection is lost byh this; however so long as b.p
  // needs access to the return of convert_ast, that's always going to be lost.
  return const_cast<smt_ast*>(conv->convert_ast(expr));
}

void
build_smt_conv_python_class(void)
{
  using namespace boost::python;

  scope solve = class_<dummy_solver_class>("solve");

  // It would appear people _do_ want to write sovlers from python.

  enum_<smt_sort_kind>("smt_sort_kind")
    .value("int", SMT_SORT_INT)
    .value("real", SMT_SORT_REAL)
    .value("bv", SMT_SORT_BV)
    .value("array", SMT_SORT_ARRAY)
    .value("bool", SMT_SORT_BOOL)
    .value("struct", SMT_SORT_STRUCT)
    .value("union", SMT_SORT_UNION);

  enum_<smt_func_kind>("smt_func_kind")
    .value("hacks", SMT_FUNC_HACKS) // Not really to be used
    .value("invalid", SMT_FUNC_INVALID) // see above
    .value("int", SMT_FUNC_INT)
    .value("bool", SMT_FUNC_BOOL)
    .value("bvint", SMT_FUNC_BVINT)
    .value("real", SMT_FUNC_REAL)
    .value("symbol", SMT_FUNC_SYMBOL)
    .value("add", SMT_FUNC_ADD)
    .value("bvadd", SMT_FUNC_BVADD)
    .value("sub", SMT_FUNC_SUB)
    .value("bvsub", SMT_FUNC_BVSUB)
    .value("mul", SMT_FUNC_MUL)
    .value("bvmul", SMT_FUNC_BVMUL)
    .value("div", SMT_FUNC_DIV)
    .value("bvudiv", SMT_FUNC_BVUDIV)
    .value("bvsdiv", SMT_FUNC_BVSDIV)
    .value("mod", SMT_FUNC_MOD)
    .value("bvsmod", SMT_FUNC_BVSMOD)
    .value("bvumod", SMT_FUNC_BVUMOD)
    .value("shl", SMT_FUNC_SHL)
    .value("bvshl", SMT_FUNC_BVSHL)
    .value("bvashr", SMT_FUNC_BVASHR)
    .value("neg", SMT_FUNC_NEG)
    .value("bvneg", SMT_FUNC_BVNEG)
    .value("bvlshr", SMT_FUNC_BVLSHR)
    .value("bvnot", SMT_FUNC_BVNOT)
    .value("bvnxor", SMT_FUNC_BVNXOR)
    .value("bvnor", SMT_FUNC_BVNOR)
    .value("bvnand", SMT_FUNC_BVNAND)
    .value("bvxor", SMT_FUNC_BVXOR)
    .value("bvor", SMT_FUNC_BVOR)
    .value("bvand", SMT_FUNC_BVAND)
    .value("implies", SMT_FUNC_IMPLIES)
    .value("xor", SMT_FUNC_XOR)
    .value("or", SMT_FUNC_OR)
    .value("and", SMT_FUNC_AND)
    .value("not", SMT_FUNC_NOT)
    .value("lt", SMT_FUNC_LT)
    .value("bvslt", SMT_FUNC_BVSLT)
    .value("bvult", SMT_FUNC_BVULT)
    .value("gt", SMT_FUNC_GT)
    .value("bvsgt", SMT_FUNC_BVSGT)
    .value("bvugt", SMT_FUNC_BVUGT)
    .value("lte", SMT_FUNC_LTE)
    .value("bvslte", SMT_FUNC_BVSLTE)
    .value("bvulte", SMT_FUNC_BVULTE)
    .value("gte", SMT_FUNC_GTE)
    .value("bvsgte", SMT_FUNC_BVSGTE)
    .value("bvugte", SMT_FUNC_BVUGTE)
    .value("eq", SMT_FUNC_EQ)
    .value("noteq", SMT_FUNC_NOTEQ)
    .value("ite", SMT_FUNC_ITE)
    .value("store", SMT_FUNC_STORE)
    .value("select", SMT_FUNC_SELECT)
    .value("concat", SMT_FUNC_CONCAT)
    .value("extract", SMT_FUNC_EXTRACT)
    .value("int2real", SMT_FUNC_INT2REAL)
    .value("real2int", SMT_FUNC_REAL2INT)
    .value("isint", SMT_FUNC_IS_INT);

  class_<smt_sort_wrapper>("smt_sort", init<smt_sort_kind>())
    .def(init<smt_sort_kind, unsigned long>())
    .def(init<smt_sort_kind, unsigned long, unsigned long>())
    .def_readwrite("id", &smt_sort::id)
    .def_readwrite("data_width", &smt_sort::data_width)
    .def_readwrite("domain_width", &smt_sort::domain_width);

  // Declare smt_ast class, wrapped, with overrides available. Note that these
  // are all declared to return internal references: by default is's the solvers
  // problem to memory manage. If overridden, then it's the overriders problem
  // to keep a python reference to all smt_ast's that C++ might point at.
  typedef return_internal_reference<> rte;
  class_<smt_ast_wrapper>("smt_ast", init<smt_convt*, smt_sortt>())
    .def_readwrite("sort", &smt_ast::sort)
    .def("ite", &smt_ast_wrapper::ite, &smt_ast_wrapper::default_ite, rte())
    .def("eq", &smt_ast_wrapper::eq, &smt_ast_wrapper::default_eq, rte())
    .def("assign", &smt_ast_wrapper::assign, &smt_ast_wrapper::default_assign)
    .def("update", &smt_ast_wrapper::update, &smt_ast_wrapper::default_update, rte())
    .def("select", &smt_ast_wrapper::select, &smt_ast_wrapper::default_select, rte())
    .def("project", &smt_ast_wrapper::project, &smt_ast_wrapper::default_project, rte());

  // Register generic smt_convt facilities: only allow the python user to do
  // expression conversion. Any new smt_convt implementation should be done
  // in C++ for example.
  // Refrain from registering enums too: basic implementation pls.
  typedef return_value_policy<return_opaque_pointer> ropaque;
  class_<smt_convt_wrapper, boost::noncopyable>("smt_convt", init<bool,const namespacet &, bool, bool, bool>())
    .def("smt_post_init", &smt_convt::smt_post_init)
    .def("convert_sort", &smt_convt::convert_sort, rte())
    .def("convert_ast", &bounce_convert_ast, rte())
    .def("push_ctx", &smt_convt::push_ctx)
    .def("pop_ctx", &smt_convt::pop_ctx)
    .def("convert_assign", &smt_convt::convert_assign, ropaque())
    .def("make_disjunct", &smt_convt::make_disjunct, ropaque())
    .def("make_conjunct", &smt_convt::make_conjunct, ropaque())
    .def("invert_ast", &smt_convt::invert_ast, ropaque())
    .def("imply_ast", &smt_convt::imply_ast, ropaque())
    .def("assert_ast", &smt_convt::assert_ast)
    .def("dec_solve", &smt_convt::dec_solve)
    .def("get", &smt_convt::get)
    // Funcs to be overridden by an extender. Same ptr ownership rules apply
    .def("mk_func_app", pure_virtual(&smt_convt_wrapper::mk_func_app_remangled), rte())
    .def("assert_ast", pure_virtual(&smt_convt::assert_ast))
    .def("dec_solve", pure_virtual(&smt_convt::dec_solve))
    .def("l_get", pure_virtual(&smt_convt::l_get))
    // Boost.python can't cope with variardic funcs, so work around it
    .def("mk_sort", pure_virtual(&smt_convt_wrapper::mk_sort_remangled), rte())
    .def("mk_smt_int", pure_virtual(&smt_convt::mk_smt_int), rte())
    .def("mk_smt_bool", pure_virtual(&smt_convt::mk_smt_bool), rte())
    .def("mk_smt_symbol", pure_virtual(&smt_convt::mk_smt_symbol), rte())
    .def("mk_smt_real", pure_virtual(&smt_convt::mk_smt_real), rte())
    .def("mk_smt_bvint", pure_virtual(&smt_convt::mk_smt_bvint), rte())
    .def("get_bool", pure_virtual(&smt_convt::get_bool))
    .def("get_bv", pure_virtual(&smt_convt::get_bv))
    .def("mk_extract", pure_virtual(&smt_convt::mk_extract), rte());

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
  for (unsigned int i = 0; i < esbmc_num_solvers; i++) {
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
    cp->def("make", &bounce_solver_factory,
        (arg("is_cpp"), arg("int_encoding"), arg("ns"), arg("options"), arg("name")=solver_name),
        return_value_policy<manage_new_object>());

    cp->staticmethod("make");
  }
}
#endif
