#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <functional>

#include <solvers/smt/smt_conv.h>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "solve.h"

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_ast)
BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_sort)

class dummy_solver_class { };

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
  opaque<smt_ast>();
  opaque<smt_sort>();

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
  scope solve2 = class_<dummy_solver_class>("solvers");
  for (unsigned int i = 0; i < esbmc_num_solvers; i++) {
    std::stringstream ss;
    const std::string &solver_name = esbmc_solvers[i].name;

    scope solve = class_<dummy_solver_class>(solver_name.c_str());

    // Trolpocolypse: we don't have a static function to create each solver,
    // or at least not one that doesn't involve mangling tuple_apis and the
    // like. So: use one function and tell python it has a default argument.
    // Use that to work out what solver to create. A possible alternative would
    // be to store a python object holding the solver name, and define a method
    // to construct it or something.
    def("make", &bounce_solver_factory,
        (arg("is_cpp"), arg("int_encoding"), arg("ns"), arg("options"), arg("name")=solver_name),
        return_value_policy<manage_new_object>());
  }
}
#endif
