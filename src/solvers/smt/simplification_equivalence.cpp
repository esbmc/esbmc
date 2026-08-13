#include <solvers/smt/simplification_equivalence.h>

#include <cstdlib>
#include <irep2/irep2_utils.h>
#include <irep2/simplification_check.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_result.h>
#include <solvers/solve.h>
#include <util/message/format.h>
#include <util/message/message.h>

namespace
{
/** Equality of these is not a plain SMT question: pointers carry an
 *  object/offset encoding whose identity is address-space state rather than a
 *  value, side effects are not expressions at all, and code has no sort. */
bool is_checkable_type(const type2tc &type)
{
  return is_bv_type(type) || is_bool_type(type) || is_fixedbv_type(type) ||
         is_floatbv_type(type);
}

bool has_unsupported_subexpr(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return true;

  if (
    is_sideeffect2t(expr) || is_dereference2t(expr) || is_address_of2t(expr) ||
    is_pointer_type(expr->type) || is_code_type(expr->type))
    return true;

  bool bad = false;
  expr->foreach_operand([&bad](const expr2tc &e) {
    if (!is_nil_expr(e) && has_unsupported_subexpr(e))
      bad = true;
  });
  return bad;
}
} // namespace

simplification_equivalencet check_simplification_equivalence(
  const expr2tc &before,
  const expr2tc &after,
  const namespacet &ns,
  const optionst &options)
{
  if (is_nil_expr(before) || is_nil_expr(after))
    return simplification_equivalencet::skipped;

  // A rewrite that changes the type is not a value-preserving claim we can
  // state as an equality; the simplifier is not supposed to make them.
  if (before->type != after->type || !is_checkable_type(before->type))
    return simplification_equivalencet::skipped;

  if (has_unsupported_subexpr(before) || has_unsupported_subexpr(after))
    return simplification_equivalencet::skipped;

  try
  {
    std::unique_ptr<smt_convt> ctx(create_solver("", ns, options));
    ctx->assert_expr(not2tc(equality2tc(before, after)));

    switch (ctx->dec_solve())
    {
    case P_UNSATISFIABLE:
      return simplification_equivalencet::equivalent;
    case P_SATISFIABLE:
      return simplification_equivalencet::differs;
    default:
      return simplification_equivalencet::skipped;
    }
  }
  catch (...)
  {
    // Conversion rejects a shape by throwing; that is a decline, not a bug in
    // the rewrite.
    return simplification_equivalencet::skipped;
  }
}

void install_simplification_equivalence_check(
  const namespacet &ns,
  const optionst &options)
{
#ifdef ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK
  // By value: namespacet is a thin handle on the context and optionst is a
  // map, and the checker outlives whatever scope installed it.
  simplification_check::install(
    [ns, options](const expr2tc &before, const expr2tc &after) {
      if (
        check_simplification_equivalence(before, after, ns, options) !=
        simplification_equivalencet::differs)
        return;

      log_error(
        "simplifier changed the meaning of an expression\n  before: {}\n  "
        "after:  {}",
        *before,
        *after);
      abort();
    });
#else
  (void)ns;
  (void)options;
#endif
}
