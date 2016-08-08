#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <functional>

#include <solvers/smt/smt_conv.h>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <util/bp_opaque_ptr.h>

#include "solve.h"

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_ast)
BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_sort)

class dummy_solver_class { };
class dummy_solver_class2 { };
class dummy_solver_class3 { };

class smt_convt_wrapper : public smt_convt, public boost::python::wrapper<smt_convt>
{
public:
  template <typename ...Args>
  smt_convt_wrapper(Args ...args) : smt_convt(args...) { }

  void
  assert_ast(smt_astt a)
  {
    this->get_override("assert_ast")(a);
  }

  resultt
  dec_solve()
  {
    return this->get_override("dec_solve")();
  }

  const std::string
  solver_text()
  {
    return this->get_override("solver_text")();
  }

  tvt
  l_get(smt_astt a)
  {
    return this->get_override("l_get")(a);
  }

  smt_sortt
  mk_sort(const smt_sort_kind k, ...)
  {
    using namespace boost::python;
    // Because this function is variadic (haha poor design choices) we can't
    // just funnel it to python. Extract actual args, then call an overrider.
    va_list ap;
    unsigned long uint;

    std::vector<object> args;
    args.push_back(object(k));

    va_start(ap, k);
    switch (k) {
    case SMT_SORT_INT:
    case SMT_SORT_REAL:
    case SMT_SORT_BOOL:
      break;
    case SMT_SORT_BV:
      uint = va_arg(ap, unsigned long);
      args.push_back(object(uint));
      break;
    case SMT_SORT_ARRAY:
    {
      smt_sort *dom = va_arg(ap, smt_sort *); // Consider constness?
      smt_sort *range = va_arg(ap, smt_sort *);
      assert(int_encoding || dom->data_width != 0);

      // XXX: setting data_width to 1 if non-bv type?
      args.push_back(object(dom));
      args.push_back(object(range));
      // XXX: how are those types going to be convertged to python references eh
    }
    default:
      std::cerr << "Unexpected sort kind " << k << " in smt_convt_wrapper mk_sort" << std::endl;
      abort();
    }

    return mk_sort_remangled(tuple(args));
  }

  smt_sortt
  mk_sort_remangled(boost::python::object o)
  {
    return this->get_override("mk_sort")(o);
  }

  smt_astt
  mk_smt_int(const mp_integer &theint, bool sign)
  {
    return this->get_override("mk_smt_int")(theint, sign);
  }

  smt_astt
  mk_smt_bool(bool val)
  {
    return this->get_override("mk_smt_bool")(val);
  }

  smt_astt
  mk_smt_symbol(const std::string &name, smt_sortt s)
  {
    return this->get_override("mk_smt_symbol")(name, s);
  }

  expr2tc
  get_bool(smt_astt a)
  {
    return this->get_override("get_bool")(a);
  }

  expr2tc
  get_bv(const type2tc &t, smt_astt a)
  {
    return this->get_override("get_bv")(t, a);
  }

  smt_astt
  mk_extract(smt_astt a, unsigned int high, unsigned int low, smt_sortt s)
  {
    return this->get_override("mk_extract")(a, high, low, s);
  }
};

class smt_ast_wrapper : public smt_ast, public boost::python::wrapper<smt_ast>
{
public:
  template <typename ...Args>
  smt_ast_wrapper(Args ...args) : smt_ast(args...) { }

  smt_astt
  ite(smt_convt *ctx, smt_astt cond, smt_astt falseop) const
  {
    using namespace boost::python;
    if (override f = this->get_override("ite"))
      return f(ctx, cond, falseop);
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
    if (override f = this->get_override("eq"))
      return f(ctx, other);
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
    if (override f = this->get_override("assign"))
      f(ctx, sym);
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
    if (override f = this->get_override("update"))
      return f(ctx, value, idx, idx_expr);
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
    if (override f = this->get_override("select"))
      return f(ctx, idx);
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
    if (override f = this->get_override("project"))
      return f(ctx, elem);
    else
      return smt_ast::project(ctx, elem);
  }

  smt_astt
  default_project(smt_convt *ctx, unsigned int elem) const
  {
    return smt_ast::project(ctx, elem);
  }
};

static smt_convt *
bounce_solver_factory(bool is_cpp, bool int_encoding, const namespacet &ns,
    const optionst &options, const char *name = "bees")
{
  std::string foo(name);
  return create_solver_factory(name, is_cpp, int_encoding, ns, options);
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

  class_<smt_sort>("smt_sort", init<smt_sort_kind>())
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
  class_<smt_convt_wrapper, boost::noncopyable>("smt_convt", no_init)
    .def("push_ctx", &smt_convt::push_ctx)
    .def("pop_ctx", &smt_convt::pop_ctx)
    .def("convert_ast", &smt_convt::convert_ast, ropaque())
    .def("convert_assign", &smt_convt::convert_assign, ropaque())
    .def("make_disjunct", &smt_convt::make_disjunct, ropaque())
    .def("make_conjunct", &smt_convt::make_conjunct, ropaque())
    .def("invert_ast", &smt_convt::invert_ast, ropaque())
    .def("imply_ast", &smt_convt::imply_ast, ropaque())
    .def("assert_ast", &smt_convt::assert_ast)
    .def("dec_solve", &smt_convt::dec_solve)
    .def("get", &smt_convt::get)
    // Funcs to be overridden by an extender. Same ptr ownership rules apply
    .def("assert_ast", pure_virtual(&smt_convt::assert_ast))
    .def("dec_solve", pure_virtual(&smt_convt::dec_solve))
    .def("l_get", pure_virtual(&smt_convt::l_get))
    // Boost.python can't cope with variardic funcs, so work around it
    .def("mk_sort", pure_virtual(&smt_convt_wrapper::mk_sort_remangled), rte())
    .def("mk_smt_int", pure_virtual(&smt_convt::mk_smt_int), rte())
    .def("mk_smt_bool", pure_virtual(&smt_convt::mk_smt_bool), rte())
    .def("mk_smt_symbol", pure_virtual(&smt_convt::mk_smt_symbol), rte())
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
