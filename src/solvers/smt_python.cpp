#ifdef WITH_PYTHON
// Don't compile... anything, otherwise.

#include <solvers/smt/smt_conv.h>
#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_ast)
BOOST_PYTHON_OPAQUE_SPECIALIZED_TYPE_ID(smt_sort)

class dummy_solver_class { };

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
}
#endif
