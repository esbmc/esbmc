#ifndef _ESBMC_PROP_SMT_SMT_CONV_H_
#define _ESBMC_PROP_SMT_SMT_CONV_H_

#include <memory>
#include <string>
#include <irep2/irep2.h>
#include <util/threeval.h>
#include <solvers/smt/smt_result.h>

// Forward declaration only: the full solver implementation lives in
// smt_solver.h and is deliberately NOT pulled in here, so that including
// smt_conv.h does not expose smt_astt / smt_sortt / ast_vec or the AST/sort
// construction machinery to the rest of ESBMC.
class smt_solver_baset;

/** Slim, frontend-facing SMT interface.
 *
 *  smt_convt owns a full solver implementation (smt_solver_baset) and exposes
 *  only the expr2tc-level API that the rest of ESBMC needs: convert an SSA
 *  equation, solve, and read back model values. The solver-handle types
 *  (smt_astt, smt_sortt, ast_vec) and the AST/sort construction machinery stay
 *  inside smt_solver_baset and are not reachable through this header.
 *
 *  Callers that genuinely need the implementation (the equation conversion
 *  loop, runtime_encoded_equationt) reach it through solver(); this is a
 *  transitional escape hatch while the conversion loop still lives in
 *  goto-symex. */
class smt_convt
{
public:
  explicit smt_convt(std::unique_ptr<smt_solver_baset> impl);
  ~smt_convt();

  /** The owned solver implementation. Transitional escape hatch for callers
   *  that still drive conversion directly (equation convert loop). */
  smt_solver_baset &solver();

  // --- Minimal external API, forwarded to the implementation ---

  void push_ctx();
  void pop_ctx();
  smt_resultt dec_solve();
  void pre_solve();
  const std::string solver_text();

  /** Fetch the model value of an expression. */
  expr2tc get(const expr2tc &expr);
  expr2tc get_by_type(const expr2tc &expr);
  expr2tc get_by_ast(const expr2tc &expr);

  /** Boolean model value of an expression. */
  tvt l_get(const expr2tc &expr);

  /** Assert a boolean expression into the solver context. */
  void assert_expr(const expr2tc &e);

  /** Convert and dump an expression in SMT format (--ssa-smt-trace). */
  void dump_expr(const expr2tc &expr);
  std::string dump_smt();
  void print_model();

private:
  std::unique_ptr<smt_solver_baset> solver_impl;
};

#endif /* _ESBMC_PROP_SMT_SMT_CONV_H_ */
