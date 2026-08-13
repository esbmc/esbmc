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
    if (!bad && !is_nil_expr(e) && has_unsupported_subexpr(e))
      bad = true;
  });
  return bad;
}

/** "The simplifier preserved this value" as an SMT predicate.
 *
 *  For floats that is not `equality2t`: it lowers to `fp.eq`, under which
 *  NaN != NaN and +0.0 == -0.0. The first makes every float rewrite -- even
 *  `x -> x` -- satisfiably unequal, and the second hides exactly the
 *  signed-zero-breaking rewrites the simplifier guards against by hand (see
 *  the abs/select rule in expr_simplifier.cpp). Bit equality via
 *  `fp.to_ieee_bv` is not an option either: SMT-LIB leaves the NaN pattern it
 *  returns unconstrained (esbmc/esbmc#6922). So state it directly -- both NaN,
 *  or equal with the same sign. */
expr2tc preserves_value(const expr2tc &before, const expr2tc &after)
{
  if (!is_floatbv_type(before->type))
    return equality2tc(before, after);

  return or2tc(
    and2tc(isnan2tc(before), isnan2tc(after)),
    and2tc(
      equality2tc(before, after),
      equality2tc(signbit2tc(before), signbit2tc(after))));
}

/** The checker's own solver options.
 *
 *  Deliberately not the run's: under --ir/--ir-ieee arithmetic is Int/Real
 *  with no wraparound, so a rewrite that is unsound for machine semantics
 *  (x*2/2 -> x) verifies as equivalent, and --smtlib makes dec_solve return
 *  P_SMTLIB, silently declining every check. The simplifier's rewrites have to
 *  hold for the machine, so always ask under bitvector semantics. */
optionst checker_options(const optionst &run_options)
{
  optionst opts = run_options;
  for (const char *opt : {"int-encoding", "ir", "ir-ieee", "smtlib", "output"})
    opts.set_option(opt, "");
  opts.set_option("fixedbv", false);
  return opts;
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
    std::unique_ptr<smt_convt> ctx(
      create_solver("", ns, checker_options(options)));
    ctx->assert_expr(not2tc(preserves_value(before, after)));

    switch (ctx->dec_solve())
    {
    case P_UNSATISFIABLE:
      return simplification_equivalencet::equivalent;
    case P_SATISFIABLE:
      return simplification_equivalencet::differs;
    default:
      log_warning("simplifier equivalence check: solver gave no verdict");
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
      switch (check_simplification_equivalence(before, after, ns, options))
      {
      case simplification_equivalencet::equivalent:
        ++simplification_check_stats::proved;
        return;

      case simplification_equivalencet::skipped:
        ++simplification_check_stats::declined;
        return;

      case simplification_equivalencet::differs:
        break;
      }

      log_error(
        "simplifier changed the meaning of an expression\n  before: {}\n  "
        "after:  {}",
        *before,
        *after);
      // Not abort(): it skips the stream flush, and this diagnostic is the
      // entire point of the run.
      exit(1);
    });
#else
  (void)ns;
  (void)options;
#endif
}

namespace simplification_check_stats
{
std::atomic<unsigned long> proved{0};
std::atomic<unsigned long> declined{0};

void report()
{
  const unsigned long p = proved.load();
  const unsigned long d = declined.load();
  if (p || d)
    log_status(
      "simplifier equivalence check: {} rewrites proved, {} declined", p, d);
}
} // namespace simplification_check_stats
