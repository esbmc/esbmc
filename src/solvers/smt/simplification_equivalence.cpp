#include <solvers/smt/simplification_equivalence.h>

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
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

simplification_equivalence_checkert::simplification_equivalence_checkert(
  const namespacet &_ns,
  const optionst &_options)
  : ns(_ns), options(checker_options(_options))
{
}

simplification_equivalence_checkert::~simplification_equivalence_checkert() =
  default;

namespace
{
void collect_symbols(const expr2tc &e, std::set<expr2tc> &out)
{
  if (is_nil_expr(e))
    return;
  if (is_symbol2t(e))
    out.insert(e);
  e->foreach_operand([&out](const expr2tc &o) { collect_symbols(o, out); });
}
} // namespace

simplification_equivalencet simplification_equivalence_checkert::check(
  const expr2tc &before,
  const expr2tc &after,
  std::string *witness)
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
    if (!ctx)
      ctx.reset(create_solver("", ns, options));

    ctx->push_ctx();
    ctx->assert_expr(not2tc(preserves_value(before, after)));
    const smt_resultt result = ctx->dec_solve();

    // The model is only readable while the frame that produced it is live.
    if (result == P_SATISFIABLE && witness)
    {
      std::set<expr2tc> symbols;
      collect_symbols(before, symbols);
      collect_symbols(after, symbols);

      witness->clear();
      for (const expr2tc &sym : symbols)
      {
        const expr2tc value = ctx->get(sym);
        if (is_nil_expr(value))
          continue;
        if (!witness->empty())
          *witness += ", ";
        *witness += fmt::format("{} = {}", *sym, *value);
      }
    }

    ctx->pop_ctx();

    switch (result)
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
    // the rewrite. The throw happened mid-frame, so the solver's state is no
    // longer trustworthy -- drop it and let the next check build a fresh one.
    ctx.reset();
    return simplification_equivalencet::skipped;
  }
}

simplification_equivalencet check_simplification_equivalence(
  const expr2tc &before,
  const expr2tc &after,
  const namespacet &ns,
  const optionst &options)
{
  return simplification_equivalence_checkert(ns, options).check(before, after);
}

void install_simplification_equivalence_check(
  const namespacet &ns,
  const optionst &options)
{
#ifdef ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK
  // The checker owns the solver and is shared into the lambda, so it lives
  // exactly as long as the installed callback.
  auto checker =
    std::make_shared<simplification_equivalence_checkert>(ns, options);
  simplification_check::install(
    [checker](const expr2tc &before, const expr2tc &after) {
      std::string witness;
      switch (checker->check(before, after, &witness))
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
        "after:  {}\n  where:  {}",
        *before,
        *after,
        witness.empty() ? "(no free symbols)" : witness);
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
