#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_solver.h>

smt_convt::smt_convt(std::unique_ptr<smt_solver_baset> impl)
  : solver_impl(std::move(impl))
{
}

smt_convt::~smt_convt() = default;

void smt_convt::push_ctx()
{
  solver_impl->push_ctx();
}

void smt_convt::pop_ctx()
{
  solver_impl->pop_ctx();
}

smt_resultt smt_convt::dec_solve()
{
  return solver_impl->dec_solve();
}

void smt_convt::pre_solve()
{
  solver_impl->pre_solve();
}

const std::string smt_convt::solver_text()
{
  return solver_impl->solver_text();
}

expr2tc smt_convt::get(const expr2tc &expr)
{
  return solver_impl->get(expr);
}

expr2tc smt_convt::get_by_type(const expr2tc &expr)
{
  return solver_impl->get_by_type(expr);
}

expr2tc smt_convt::get_by_ast(const expr2tc &expr)
{
  return solver_impl->get_by_ast(expr);
}

tvt smt_convt::l_get(const expr2tc &expr)
{
  return solver_impl->l_get(expr);
}

void smt_convt::assert_expr(const expr2tc &e)
{
  solver_impl->assert_expr(e);
}

void smt_convt::convert_ast(const expr2tc &expr)
{
  // Discard the handle: callers only need the expression encoded into the
  // solver, and the AST is retained in the implementation's cache.
  solver_impl->convert_ast(expr);
}

void smt_convt::convert_assign(const expr2tc &expr)
{
  solver_impl->convert_assign(expr);
}

void smt_convt::renumber_symbol_address(
  const expr2tc &guard,
  const expr2tc &addr_symbol,
  const expr2tc &new_size)
{
  solver_impl->renumber_symbol_address(guard, addr_symbol, new_size);
}

void smt_convt::dump_expr(const expr2tc &expr)
{
  solver_impl->dump_expr(expr);
}

std::string smt_convt::dump_smt()
{
  return solver_impl->dump_smt();
}

void smt_convt::print_model()
{
  solver_impl->print_model();
}
