#include "simplification_check.h"

#include <solvers/smt/smt_conv.h>
#include <solvers/solve.h>

void simplification_check::check_equivalence(
  const expr2tc &old_expr,
  const expr2tc &new_expr,
  const namespacet &ns)
{
  if (is_nil_expr(old_expr) || is_nil_expr(new_expr))
    return;
  // for (const auto& pretty : {old_expr->pretty(), new_expr->pretty()})
  // {
  //   if (
  //     pretty.find("pointer") != std::string::npos ||
  //     pretty.find("sideeffect") != std::string::npos)
  //   {
  //     return;
  //   }
  // }
  if (
    is_bv_type(old_expr) || is_fixedbv_type(old_expr) ||
    is_floatbv_type(old_expr) || is_bool_type(old_expr) ||
    is_array_type(old_expr) || is_struct_type(old_expr) ||
    is_union_type(old_expr) || is_pointer_type(old_expr))
  {
    // const contextt context;
    // const namespacet ns{context};
    smt_convt *smt_ctx = create_solver("", ns, {});
    // TODO: Consider reusing the same solver context for multiple checks
    const auto smt_old_expr = smt_ctx->convert_ast(old_expr);
    const auto smt_new_expr = smt_ctx->convert_ast(new_expr);

    // Negate the equality to check for refinement
    smt_ctx->assert_ast(
      smt_ctx->mk_not(smt_old_expr->eq(smt_ctx, smt_new_expr)));

    if (const auto result = smt_ctx->dec_solve();
        result == smt_convt::P_SATISFIABLE)
    {
      // Get concrete values for old_expr and new_expr
      auto concrete_old_expr = smt_ctx->get_by_type(old_expr);
      auto concrete_new_expr = smt_ctx->get_by_type(new_expr);

      log_error(
        "Refinement check failed. old expr:\n {}\nnew expr:\n {}\n old expr "
        "value:\n {}\nnew expr value:\n {}",
        old_expr->pretty(),
        new_expr->pretty(),
        concrete_old_expr->pretty(),
        concrete_new_expr->pretty());
      smt_ctx->print_model();
      smt_ctx->dump_smt();
      abort();
    }
    else if (result == smt_convt::P_UNSATISFIABLE)
    {
    }
    else if (result == smt_convt::P_ERROR || result == smt_convt::P_SMTLIB)
    {
      log_warning("Refinement check failed: solver error [IGNORED]");
    }
  }
}
