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

class smt_convt_wrapper : public smt_convt,  boost::python::wrapper<smt_convt>
{
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

    return this->get_override("mk_sort")(tuple(args));
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

  // There are no circumstances where python users should be inspecting the
  // internals of SMT handles. They're handles. In terms of lifetime, the smt
  // handles get destroyed when the smt_convt gets destroyed. Now, it would
  // be illegal to pass a handle from one smt_convt to another smt_convt; in
  // the same way, if your smt_convt has gone out of scope and been destroyed,
  // you don't have anywhere to legally pass these opqaue pointers.
  //
  // That rather ignores the matter of push/popping scopes. User is on their
  // own there.

  // Use vendor'd opaque as opaque_const, because the 'extract' method in
  // boost.python's version catches fire when handed a const qualification.
  opaque_const<const smt_ast>();
  opaque_const<const smt_sort>();

  // Register generic smt_convt facilities: only allow the python user to do
  // expression conversion. Any new smt_convt implementation should be done
  // in C++ for example.
  // Refrain from registering enums too: basic implementation pls.
  typedef return_value_policy<return_opaque_pointer> ropaque;
  class_<smt_convt, boost::noncopyable>("smt_convt", no_init)
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
    .def("get", &smt_convt::get);

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
