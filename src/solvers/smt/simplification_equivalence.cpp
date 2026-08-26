#include <solvers/smt/simplification_equivalence.h>

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <irep2/irep2_utils.h>
#include <irep2/simplification_check.h>
#include <solvers/smt/smt_conv.h>
#include <solvers/smt/smt_result.h>
#include <solvers/solve.h>
#include <util/arith/ieee_float.h>
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

/** The rounding mode @p expr carries, or nil where it carries none. */
const expr2tc &rounding_mode_of(const expr2tc &expr)
{
  static const expr2tc none;
  switch (expr->expr_id)
  {
  case expr2t::ieee_add_id:
    return to_ieee_add2t(expr).rounding_mode;
  case expr2t::ieee_sub_id:
    return to_ieee_sub2t(expr).rounding_mode;
  case expr2t::ieee_mul_id:
    return to_ieee_mul2t(expr).rounding_mode;
  case expr2t::ieee_div_id:
    return to_ieee_div2t(expr).rounding_mode;
  case expr2t::ieee_rem_id:
    return to_ieee_rem2t(expr).rounding_mode;
  case expr2t::ieee_fma_id:
    return to_ieee_fma2t(expr).rounding_mode;
  case expr2t::ieee_sqrt_id:
    return to_ieee_sqrt2t(expr).rounding_mode;
  case expr2t::nearbyint_id:
    return to_nearbyint2t(expr).rounding_mode;
  case expr2t::typecast_id:
    // Every typecast carries one, but conversion only consults it when the
    // result is a float; screening the rest would decline every integer cast in
    // the run.
    return is_floatbv_type(expr->type) ? to_typecast2t(expr).rounding_mode
                                       : none;
  default:
    return none;
  }
}

/** Whether every backend can build the rounding mode @p expr asks for.
 *
 *  A symbolic rounding mode makes convert_rounding_mode() build an ite over all
 *  five modes, ROUND_TO_AWAY among them, and the Z3 and CVC5 backends abort()
 *  on that one rather than returning a term -- so does a literal ROUND_TO_AWAY.
 *  abort() is not something check()'s catch can turn back into a decline, so
 *  the shape has to be refused before conversion sees it. The abort is a defect
 *  in those backends (Bitwuzla builds RNA happily) and is reachable without
 *  this check at all; declining here is not a fix for it. */
bool convertible_rounding_mode(const expr2tc &expr)
{
  const expr2tc &rm = rounding_mode_of(expr);
  if (is_nil_expr(rm))
    return true;
  return is_constant_int2t(rm) &&
         to_constant_int2t(rm).value != BigInt(ieee_floatt::ROUND_TO_AWAY);
}

/** Nodes past which a rewrite is declined rather than proved.
 *
 *  Checking per node means one query per fold, each over the whole node. On a
 *  chain of N terms the simplifier folds at every level, so the run costs O(N)
 *  queries of O(N) terms -- regression/esbmc/deep_binary_chain_fail builds a
 *  12288-term conjunction and does not finish. The peepholes a large node
 *  exercises are the same ones a small node exercises, so the assurance given
 *  up is small and the declined counter records it. */
const std::size_t max_checked_nodes = 256;

/** Whether @p expr has at most @p budget nodes, counted with an early exit so
 *  the screen cannot itself become the cost it exists to avoid. */
bool within_size_budget(const expr2tc &expr, std::size_t &budget)
{
  if (budget == 0)
    return false;
  --budget;

  bool fits = true;
  expr->foreach_operand([&](const expr2tc &e) {
    if (fits && !is_nil_expr(e))
      fits = within_size_budget(e, budget);
  });
  return fits;
}

/** Shapes conversion cannot be asked for a term, beyond the type-level screens
 *  above. Split out from has_unsupported_subexpr() to keep that function inside
 *  the complexity gate. */
bool is_unstatable_shape(const expr2tc &expr)
{
  // Value-set placeholders standing in for "symex could not say". They carry no
  // value to preserve -- and convert_ast aborts() rather than throwing on one,
  // which check()'s catch cannot rescue. Symex is where they arise, so nothing
  // met them until the check started covering it (esbmc/esbmc#7260).
  if (is_unknown2t(expr) || is_invalid2t(expr))
    return true;

  if (!convertible_rounding_mode(expr))
    return true;

  // overflow_cast()'s lowering builds its upper bound as
  // constant_int(2^bits - 1) in the *operand's* type (smt_overflow.cpp), which
  // is unrepresentable and wraps to -1 when bits equals a signed operand's
  // width. The run's own --ir encoding does not wrap, so the disagreement is
  // with the bitvector encoding this check forces, not with the fold.
  if (is_overflow_cast2t(expr))
    return true;

  // convert_terminal() asserts a fixedbv constant fits a uint64_t. --fixedbv
  // runs build wider ones (regression/esbmc/github_562), and the simplifier
  // folds them long before conversion would have rejected them.
  if (is_constant_fixedbv2t(expr) && expr->type->get_width() > 64)
    return true;

  // Byte ops over an aggregate. convert_byte_extract() asserts outright on an
  // array source, and over a struct the simplifier reasons about members while
  // conversion reasons about a flat bitvector -- the two disagree on shapes
  // this check cannot arbitrate. The pipeline lowers aggregates before
  // conversion sees them; the simplifier runs ahead of that.
  if (
    (is_byte_extract2t(expr) &&
     !is_checkable_type(to_byte_extract2t(expr).source_value->type)) ||
    (is_byte_update2t(expr) &&
     !is_checkable_type(to_byte_update2t(expr).source_value->type)))
    return true;

  // Narrowing a float to an integer has no value to preserve unless the value
  // is in range: C11 6.3.1.4p1 makes it undefined, and SMT-LIB leaves fp.to_sbv
  // and fp.to_ubv unspecified for NaN, the infinities, and anything out of
  // range. The solver may pick a different unspecified value than the
  // simplifier folded to, which is a disagreement about undefined behaviour
  // rather than about meaning.
  return is_typecast2t(expr) &&
         is_floatbv_type(to_typecast2t(expr).from->type) &&
         is_bv_type(expr->type);
}

bool has_unsupported_subexpr(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return true;

  if (
    is_sideeffect2t(expr) || is_dereference2t(expr) || is_address_of2t(expr) ||
    is_pointer_type(expr->type) || is_code_type(expr->type))
    return true;

  if (is_unstatable_shape(expr))
    return true;

  bool bad = false;
  expr->foreach_operand([&bad](const expr2tc &e) {
    if (!bad && !is_nil_expr(e) && has_unsupported_subexpr(e))
      bad = true;
  });
  return bad;
}

/** Width of the flat bitvector conversion builds for @p type, or 0 where it
 *  builds no flat bitvector: bool, and the aggregates that meet through the
 *  tuple/array flatteners instead. */
unsigned bv_width(const type2tc &type)
{
  if (is_bv_type(type) || is_fixedbv_type(type) || is_floatbv_type(type))
    return type->get_width();
  return 0;
}

/** Will @p a and @p b convert to the same sort? Equal widths are not enough:
 *  a float and a bitvector of the same width are different sorts, and only
 *  a float pair takes convert_ast()'s fp.eq path. */
bool sorts_match(const type2tc &a, const type2tc &b)
{
  return is_floatbv_type(a) == is_floatbv_type(b) && bv_width(a) == bv_width(b);
}

/** Node kinds whose two operands conversion requires to share a sort.
 *
 *  Shifts are deliberately absent: irep2 allows a shift count narrower than
 *  the value, and convert_ast() casts it up (smt_solver.cpp, shl/ashr/lshr). */
bool operands_must_share_sort(const expr2tc &expr)
{
  switch (expr->expr_id)
  {
  case expr2t::equality_id:
  case expr2t::notequal_id:
  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
  case expr2t::add_id:
  case expr2t::sub_id:
  case expr2t::mul_id:
  case expr2t::div_id:
  case expr2t::modulus_id:
  case expr2t::bitand_id:
  case expr2t::bitor_id:
  case expr2t::bitxor_id:
    return true;
  default:
    return false;
  }
}

/** Would conversion of @p expr meet the operand-sort preconditions it asserts?
 *
 *  Every backend's mk_eq, mk_ite and their relational/arithmetic siblings open
 *  with `assert(a->sort->get_data_width() == b->sort->get_data_width())`. An
 *  assert is abort(), not a throw, so an ill-sorted term takes the process
 *  down instead of the decline path in check() (esbmc/esbmc#7220).
 *
 *  This is a blacklist of the shapes known to assert, not a proof that what it
 *  passes converts: a precondition not listed here still aborts. The source of
 *  truth for the list is the assert cluster in each backend, bitwuzla_conv.cpp
 *  being the readable one -- nothing links the two, so they drift. */
bool sorts_agree(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return true;

  if (operands_must_share_sort(expr))
  {
    const expr2tc &side_1 = *expr->get_sub_expr(0);
    const expr2tc &side_2 = *expr->get_sub_expr(1);
    if (
      is_nil_expr(side_1) || is_nil_expr(side_2) ||
      !sorts_match(side_1->type, side_2->type))
      return false;
  }

  // if2t cannot join the list above: its operands are (cond, true, false), so
  // the pair mk_ite constrains is 1 and 2, not 0 and 1.
  if (is_if2t(expr))
  {
    const if2t &branch = to_if2t(expr);
    if (!sorts_match(branch.true_value->type, branch.false_value->type))
      return false;
  }

  bool agree = true;
  expr->foreach_operand([&agree](const expr2tc &e) {
    if (agree)
      agree = sorts_agree(e);
  });
  return agree;
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

namespace
{
/** Counting lives here rather than in the installed callback so that a direct
 *  caller -- the one-shot free function, the unit tests -- cannot move one
 *  counter without the others and leave the report line self-contradictory. */
simplification_equivalencet decline()
{
  ++simplification_check_stats::declined;
  return simplification_equivalencet::skipped;
}
} // namespace

void simplification_equivalence_checkert::record_witness(
  const expr2tc &before,
  const expr2tc &after,
  smt_convt &ctx,
  std::string &witness)
{
  std::set<expr2tc> symbols;
  collect_symbols(before, symbols);
  collect_symbols(after, symbols);

  witness.clear();
  for (const expr2tc &sym : symbols)
  {
    const expr2tc value = ctx.get(sym);
    if (is_nil_expr(value))
      continue;
    if (!witness.empty())
      witness += ", ";
    witness += fmt::format("{} = {}", *sym, *value);
  }
}

smt_convt &simplification_equivalence_checkert::solver()
{
  // create_solver() simplifies, but detail::run()'s reentrancy flag is
  // thread-local and already set for this thread, so it cannot recurse back in.
  std::lock_guard<std::mutex> lock(solvers_mutex);
  std::unique_ptr<smt_convt> &c = solvers[std::this_thread::get_id()];
  if (!c)
    c.reset(create_solver("", ns, options));
  return *c;
}

void simplification_equivalence_checkert::drop_solver()
{
  std::lock_guard<std::mutex> lock(solvers_mutex);
  solvers.erase(std::this_thread::get_id());
}

simplification_equivalencet simplification_equivalence_checkert::check(
  const expr2tc &before,
  const expr2tc &after,
  std::string *witness)
{
  if (is_nil_expr(before) || is_nil_expr(after))
    return decline();

  // A rewrite that changes the type is not a value-preserving claim we can
  // state as an equality; the simplifier is not supposed to make them.
  if (before->type != after->type || !is_checkable_type(before->type))
    return decline();

  if (has_unsupported_subexpr(before) || has_unsupported_subexpr(after))
    return decline();

  // One budget across both sides: what has to stay affordable is the query,
  // and the query carries both.
  std::size_t budget = max_checked_nodes;
  if (!within_size_budget(before, budget) || !within_size_budget(after, budget))
    return decline();

  // An ill-sorted `before` predates the rewrite -- ESBMC built it that way and
  // the normal pipeline never converts it in that form -- so it is a decline.
  // Screening it first is what makes an ill-sorted `after` attributable: the
  // rewrite introduced it, which is a defect whatever it did to the value.
  if (!sorts_agree(before))
  {
    ++simplification_check_stats::ill_sorted;
    return decline();
  }
  if (!sorts_agree(after))
    return simplification_equivalencet::malformed;

  try
  {
    smt_convt &ctx = solver();

    ctx.push_ctx();
    ctx.assert_expr(not2tc(preserves_value(before, after)));
    const smt_resultt result = ctx.dec_solve();

    // The model is only readable while the frame that produced it is live.
    if (result == P_SATISFIABLE && witness)
      record_witness(before, after, ctx, *witness);

    ctx.pop_ctx();

    switch (result)
    {
    case P_UNSATISFIABLE:
      ++simplification_check_stats::proved;
      return simplification_equivalencet::equivalent;
    case P_SATISFIABLE:
      return simplification_equivalencet::differs;
    default:
      log_warning("simplifier equivalence check: solver gave no verdict");
      return decline();
    }
  }
  catch (...)
  {
    // Conversion declines a shape by throwing (smt_casts.cpp); that is a
    // decline, not a bug in the rewrite. It cannot report a violated
    // precondition this way -- see sorts_agree(). The throw happened
    // mid-frame, so the solver's state is no longer trustworthy: drop it and
    // let the next check build a fresh one.
    drop_solver();
    return decline();
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
      const simplification_equivalencet verdict =
        checker->check(before, after, &witness);
      if (
        verdict == simplification_equivalencet::equivalent ||
        verdict == simplification_equivalencet::skipped)
        return;

      log_error(
        "{}\n  before: {}\n  after:  {}\n  where:  {}",
        verdict == simplification_equivalencet::differs
          ? "simplifier changed the meaning of an expression"
          : "simplifier produced an expression whose operand sorts disagree",
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

simplification_check_scopet::~simplification_check_scopet()
{
  simplification_check_stats::report();
  simplification_check::clear();
}

namespace simplification_check_stats
{
std::atomic<unsigned long> proved{0};
std::atomic<unsigned long> declined{0};
std::atomic<unsigned long> ill_sorted{0};

void report()
{
  const unsigned long p = proved.load();
  const unsigned long d = declined.load();
  const unsigned long i = ill_sorted.load();
  if (p || d)
    log_status(
      "simplifier equivalence check: {} rewrites proved, {} declined{}",
      p,
      d,
      i ? fmt::format(" ({} ill-sorted)", i) : std::string());
}
} // namespace simplification_check_stats
