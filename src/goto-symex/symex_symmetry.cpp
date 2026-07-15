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

// +1 when the comparison being true selects the greater operand (max),
// -1 when it selects the lesser (min).
int extremum_of(cmp_kindt k)
{
  return (k == cmp_kindt::GT || k == cmp_kindt::GE) ? 1 : -1;
}

// Follow at most one `def_of` hop -- enough to see through the temporary
// ESBMC's phi-merge lowering inserts around `if (cond) x = v;`, so cost
// stays O(1) per operand.
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

// Returns +1 if `ite(cond, t, e)` is a max idiom (result >= t and >= e),
// -1 for min, 0 otherwise. A pure fact about ite/comparison semantics, so
// no solver confirmation is needed.
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

    // The injected bounds are tautologies only over a total order; floatbv
    // is excluded because IEEE-754 comparisons involving NaN are all false.
    const type2tc &ty = ite.type;
    if (!(is_signedbv_type(ty) || is_unsignedbv_type(ty) ||
          is_fixedbv_type(ty) || is_pointer_type(ty)))
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
