#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <memory>
#include <solvers/smt/smt_solver.h>

/** Slim, frontend-facing SMT interface.
 *
 *  smt_convt owns a full solver implementation (smt_solver_baset) and exposes
 *  only the expr2tc-level API that the rest of ESBMC needs: convert an SSA
 *  equation, solve, and read back model values. The solver-handle types
 *  (smt_astt, smt_sortt, ast_vec) and the AST/sort construction machinery stay
 *  inside smt_solver_baset and never leak through this wrapper.
 *
 *  Callers that genuinely need the implementation (the equation conversion
 *  loop, runtime_encoded_equationt) reach it through solver(); this is a
 *  transitional escape hatch while the conversion loop still lives in
 *  goto-symex. */
class smt_convt
{
public:
  explicit smt_convt(std::unique_ptr<smt_solver_baset> impl)
    : solver_impl(std::move(impl))
  {
  }

  typedef smt_solver_baset::resultt resultt;
  static constexpr resultt P_UNSATISFIABLE = smt_solver_baset::P_UNSATISFIABLE;
  static constexpr resultt P_SATISFIABLE = smt_solver_baset::P_SATISFIABLE;
  static constexpr resultt P_ERROR = smt_solver_baset::P_ERROR;
  static constexpr resultt P_SMTLIB = smt_solver_baset::P_SMTLIB;

  /** The owned solver implementation. Transitional escape hatch for callers
   *  that still drive conversion directly (equation convert loop). */
  smt_solver_baset &solver()
  {
    return *solver_impl;
  }

  // --- Minimal external API, forwarded to the implementation ---

  void push_ctx()
  {
    solver_impl->push_ctx();
  }
  void pop_ctx()
  {
    solver_impl->pop_ctx();
  }
  resultt dec_solve()
  {
    return solver_impl->dec_solve();
  }
  void pre_solve()
  {
    solver_impl->pre_solve();
  }
  const std::string solver_text()
  {
    return solver_impl->solver_text();
  }

  /** Fetch the model value of an expression. */
  expr2tc get(const expr2tc &expr)
  {
    return solver_impl->get(expr);
  }
  expr2tc get_by_type(const expr2tc &expr)
  {
    return solver_impl->get_by_type(expr);
  }
  expr2tc get_by_ast(const expr2tc &expr)
  {
    return solver_impl->get_by_ast(expr);
  }

  /** Boolean model value of an expression. */
  tvt l_get(const expr2tc &expr)
  {
    return solver_impl->l_get(expr);
  }

  /** Assert a boolean expression into the solver context. */
  void assert_expr(const expr2tc &e)
  {
    solver_impl->assert_expr(e);
  }

  /** Convert and dump an expression in SMT format (--ssa-smt-trace). */
  void dump_expr(const expr2tc &expr)
  {
    solver_impl->dump_expr(expr);
  }
  std::string dump_smt()
  {
    return solver_impl->dump_smt();
  }
  void print_model()
  {
    solver_impl->print_model();
  }

  /** Scope guard memoising l_get() results for the duration of trace
   *  construction. Forwards to the implementation's cache flags. */
  struct model_cache_scopet
  {
    smt_solver_baset::model_cache_scopet scope;
    explicit model_cache_scopet(smt_convt &c) : scope(*c.solver_impl)
    {
    }
  };

private:
  std::unique_ptr<smt_solver_baset> solver_impl;
};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
