#include <goto-symex/symex_symmetry.h>
#include <solvers/smt/smt_conv.h>
#include <util/message.h>
#include <irep2/irep2.h>
#include <string>
#include <unordered_map>

namespace
{
enum class cmp_kindt
{
  GT,
  GE,
  LT,
  LE
};

bool as_comparison(
  const expr2tc &e,
  cmp_kindt &kind,
  expr2tc &side_1,
  expr2tc &side_2)
{
  if (is_greaterthan2t(e))
  {
    kind = cmp_kindt::GT;
    side_1 = to_greaterthan2t(e).side_1;
    side_2 = to_greaterthan2t(e).side_2;
  }
  else if (is_greaterthanequal2t(e))
  {
    kind = cmp_kindt::GE;
    side_1 = to_greaterthanequal2t(e).side_1;
    side_2 = to_greaterthanequal2t(e).side_2;
  }
  else if (is_lessthan2t(e))
  {
    kind = cmp_kindt::LT;
    side_1 = to_lessthan2t(e).side_1;
    side_2 = to_lessthan2t(e).side_2;
  }
  else if (is_lessthanequal2t(e))
  {
    kind = cmp_kindt::LE;
    side_1 = to_lessthanequal2t(e).side_1;
    side_2 = to_lessthanequal2t(e).side_2;
  }
  else
    return false;
  return true;
}

cmp_kindt flip(cmp_kindt k)
{
  switch (k)
  {
  case cmp_kindt::GT:
    return cmp_kindt::LE;
  case cmp_kindt::LE:
    return cmp_kindt::GT;
  case cmp_kindt::GE:
    return cmp_kindt::LT;
  case cmp_kindt::LT:
    return cmp_kindt::GE;
  }
  abort();
}

// +1: side_1 op side_2 being true selects the greater of the two operands
// (max idiom); -1: selects the lesser (min idiom).
int extremum_of(cmp_kindt k)
{
  return (k == cmp_kindt::GT || k == cmp_kindt::GE) ? 1 : -1;
}

// One optional resolution hop through `def_of`. Used only to see through
// the separate guard/branch-value SSA steps that ESBMC's phi-merge lowering
// introduces around an `if (cond) x = v;` statement (the ite's own operand
// is a plain symbol reference to a temporary defined by a preceding step),
// not to inline arbitrarily deep -- so cost stays O(1) per operand.
expr2tc resolve_once(
  const expr2tc &e,
  const std::unordered_map<std::string, expr2tc> &def_of)
{
  if (!is_symbol2t(e))
    return e;
  auto it = def_of.find(to_symbol2t(e).get_symbol_name());
  return it == def_of.end() ? e : it->second;
}

bool operand_matches(
  const expr2tc &cond_side,
  const expr2tc &branch,
  const std::unordered_map<std::string, expr2tc> &def_of)
{
  return cond_side == branch || cond_side == resolve_once(branch, def_of);
}

// Recognises `ite(cond, t, e)` as a max/min idiom: `cond` (after peeling at
// most one leading `not` and one symbol-lookup hop) is a comparison whose
// two operands are `t` and `e` in either order. Returns +1 for max (the
// ite's result is always >= t and >= e), -1 for min (<= t and <= e), 0 if
// `cond` doesn't match this shape. This is a fact about ite/comparison
// semantics alone -- true regardless of what t, e, or cond mean -- so no
// solver confirmation is needed.
int classify_ite_extremum(
  expr2tc cond,
  const expr2tc &t,
  const expr2tc &e,
  const std::unordered_map<std::string, expr2tc> &def_of)
{
  bool negated = is_not2t(cond);
  if (negated)
    cond = to_not2t(cond).value;
  if (is_symbol2t(cond))
    cond = resolve_once(cond, def_of);

  cmp_kindt kind;
  expr2tc side_1, side_2;
  if (!as_comparison(cond, kind, side_1, side_2))
    return 0;
  if (negated)
    kind = flip(kind);

  if (operand_matches(side_1, t, def_of) && operand_matches(side_2, e, def_of))
    return extremum_of(kind);
  if (operand_matches(side_1, e, def_of) && operand_matches(side_2, t, def_of))
    return -extremum_of(kind);

  return 0;
}
} // namespace

void assert_symmetry_breaking(
  const symex_target_equationt &equation,
  smt_convt &smt_conv)
{
  std::unordered_map<std::string, expr2tc> def_of;
  for (const auto &step : equation.SSA_steps)
  {
    if (step.ignore || !step.is_assignment() || !is_symbol2t(step.lhs))
      continue;
    def_of[to_symbol2t(step.lhs).get_symbol_name()] = step.rhs;
  }

  unsigned hints = 0;
  for (const auto &step : equation.SSA_steps)
  {
    if (step.ignore || !step.is_assignment() || !is_symbol2t(step.lhs))
      continue;
    if (!is_if2t(step.rhs))
      continue;

    const if2t &ite = to_if2t(step.rhs);

    // IEEE-754 comparisons are not a total order (any relation involving a
    // NaN operand is false), so `ite(t > e, t, e)` is not provably >= t and
    // >= e when t or e may be NaN -- the soundness argument below only
    // holds for total-order comparisons (int/bitvector/pointer).
    if (is_floatbv_type(ite.type))
      continue;

    int dir =
      classify_ite_extremum(ite.cond, ite.true_value, ite.false_value, def_of);
    if (dir == 0)
      continue;

    smt_conv.assert_expr(
      dir > 0 ? lessthanequal2tc(ite.true_value, step.lhs)
              : lessthanequal2tc(step.lhs, ite.true_value));
    smt_conv.assert_expr(
      dir > 0 ? lessthanequal2tc(ite.false_value, step.lhs)
              : lessthanequal2tc(step.lhs, ite.false_value));
    hints += 2;
  }

  if (hints > 0)
    log_debug(
      "symmetry-breaking",
      "injected {} redundant extremum bound(s) from recognised max/min ite "
      "assignments",
      hints);
}
