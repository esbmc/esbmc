#include <goto-symex/symex_symmetry.h>
#include <util/message.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace
{
using SSA_stepst = symex_target_equationt::SSA_stepst;

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

// A running max/min fold still being extended: `tip` is the latest result,
// `tip_step` is the assignment that defines it (so a direct leaf-to-final
// bound can be injected right after it), and `leaves` are all the original
// (non-chain) values folded into it so far.
struct chain_infot
{
  expr2tc tip;
  SSA_stepst::iterator tip_step;
  std::vector<expr2tc> leaves;
  int dir;
  unsigned depth;
};

// Looks up `operand` as another chain's current tip, resolving one def_of hop
// first (mirrors operand_matches) so phi-merge temporaries don't hide it.
chain_infot *find_tip(
  std::unordered_map<std::string, chain_infot> &active_tip,
  const expr2tc &operand,
  int dir,
  const std::unordered_map<std::string, expr2tc> &def_of)
{
  if (!is_symbol2t(operand))
    return nullptr;

  auto try_name = [&](const std::string &name) -> chain_infot * {
    auto it = active_tip.find(name);
    return it != active_tip.end() && it->second.dir == dir ? &it->second
                                                           : nullptr;
  };

  if (chain_infot *hit = try_name(to_symbol2t(operand).get_symbol_name()))
    return hit;

  expr2tc resolved = resolve_once(operand, def_of);
  return is_symbol2t(resolved) && resolved != operand
           ? try_name(to_symbol2t(resolved).get_symbol_name())
           : nullptr;
}

// Extends (or starts) the running chain(s) that `lhs = ite(cond, t, e)`
// (defined at `lhs_step`) belongs to, so the leaf-to-final bounds can be
// injected in one pass over `active_tip` once the SSA scan is done, instead
// of re-deriving them via solver-side transitive chaining across every
// intermediate step.
void extend_chain(
  std::unordered_map<std::string, chain_infot> &active_tip,
  const expr2tc &lhs,
  SSA_stepst::iterator lhs_step,
  const expr2tc &t,
  const expr2tc &e,
  int dir,
  const std::unordered_map<std::string, expr2tc> &def_of)
{
  chain_infot *t_tip = find_tip(active_tip, t, dir, def_of);
  chain_infot *e_tip = find_tip(active_tip, e, dir, def_of);

  chain_infot next;
  next.tip = lhs;
  next.tip_step = lhs_step;
  next.dir = dir;

  if (t_tip && e_tip && t_tip == e_tip)
  {
    // Both branches resolve to the same active tip (e.g. ite(cond, x, x)):
    // this fold introduces no new leaf, just renames the tip to `lhs`.
    next.leaves = std::move(t_tip->leaves);
    next.depth = t_tip->depth;
  }
  else if (t_tip && e_tip)
  {
    next.leaves = std::move(t_tip->leaves);
    next.leaves.insert(
      next.leaves.end(),
      std::make_move_iterator(e_tip->leaves.begin()),
      std::make_move_iterator(e_tip->leaves.end()));
    next.depth = t_tip->depth + e_tip->depth;
  }
  else if (t_tip)
  {
    next.leaves = std::move(t_tip->leaves);
    next.leaves.push_back(e);
    next.depth = t_tip->depth + 1;
  }
  else if (e_tip)
  {
    next.leaves = std::move(e_tip->leaves);
    next.leaves.push_back(t);
    next.depth = e_tip->depth + 1;
  }
  else
  {
    next.leaves = {t, e};
    next.depth = 1;
  }

  if (t_tip)
    active_tip.erase(to_symbol2t(t_tip->tip).get_symbol_name());
  if (e_tip && e_tip != t_tip)
    active_tip.erase(to_symbol2t(e_tip->tip).get_symbol_name());
  active_tip[to_symbol2t(lhs).get_symbol_name()] = std::move(next);
}

// Builds an unconditional ASSUME step carrying `cond`, borrowing the source
// location and loop number of the assignment it was derived from.
symex_target_equationt::SSA_stept make_assume(
  const expr2tc &cond,
  const symex_target_equationt::SSA_stept &origin)
{
  symex_target_equationt::SSA_stept step;
  step.type = goto_trace_stept::ASSUME;
  step.guard = gen_true_expr();
  step.cond = cond;
  step.source = origin.source;
  step.loop_number = origin.loop_number;
  return step;
}
} // namespace

// Precondition: called once per freshly produced equation. The pass holds no
// cross-run state, but it is not idempotent -- re-running it over a list that
// already carries its ASSUME steps would inject a duplicate (harmless but
// redundant) set of bounds. The BMC driver satisfies this: each formula from
// get_next_formula() gets its own algorithms run.
bool symmetry_breakingt::run(SSA_stepst &steps)
{
  std::unordered_map<std::string, expr2tc> def_of;
  std::unordered_map<std::string, chain_infot> active_tip;

  // Bounds are collected against the assignment they follow and inserted
  // once the scan is done: mutating the list mid-scan would feed the new
  // ASSUME steps back into the loop, and list iterators stay valid across
  // insertions so the recorded positions remain sound.
  std::vector<std::pair<SSA_stepst::iterator, expr2tc>> pending;

  unsigned per_step_hints = 0;
  unsigned direct_hints = 0;
  for (auto it = steps.begin(); it != steps.end(); ++it)
  {
    const symex_target_equationt::SSA_stept &step = *it;
    if (step.ignore || !step.is_assignment() || !is_symbol2t(step.lhs))
      continue;

    if (is_if2t(step.rhs))
    {
      const if2t &ite = to_if2t(step.rhs);

      // The injected bounds are tautologies only over a total order;
      // floatbv is excluded because IEEE-754 comparisons involving NaN are
      // all false.
      const type2tc &ty = ite.type;
      if (
        is_signedbv_type(ty) || is_unsignedbv_type(ty) || is_fixedbv_type(ty) ||
        is_pointer_type(ty))
      {
        int dir = classify_ite_extremum(
          ite.cond, ite.true_value, ite.false_value, def_of);
        if (dir != 0)
        {
          pending.emplace_back(
            it,
            dir > 0 ? lessthanequal2tc(ite.true_value, step.lhs)
                    : lessthanequal2tc(step.lhs, ite.true_value));
          pending.emplace_back(
            it,
            dir > 0 ? lessthanequal2tc(ite.false_value, step.lhs)
                    : lessthanequal2tc(step.lhs, ite.false_value));
          per_step_hints += 2;

          extend_chain(
            active_tip,
            step.lhs,
            it,
            ite.true_value,
            ite.false_value,
            dir,
            def_of);
        }
      }
    }

    def_of[to_symbol2t(step.lhs).get_symbol_name()] = step.rhs;
  }

  // Whatever chain tips are still unconsumed are chain-finals: inject a
  // direct bound from every leaf ever folded into them, so proving a
  // property against the final result doesn't need the solver to chain
  // per-step bounds transitively across the whole fold. Chains of depth 1
  // are skipped since their leaves are exactly the per-step bounds above.
  for (const auto &[name, info] : active_tip)
  {
    if (info.depth < 2)
      continue;
    for (const expr2tc &leaf : info.leaves)
    {
      pending.emplace_back(
        info.tip_step,
        info.dir > 0 ? lessthanequal2tc(leaf, info.tip)
                     : lessthanequal2tc(info.tip, leaf));
      direct_hints++;
    }
  }

  for (const auto &[pos, cond] : pending)
    steps.insert(std::next(pos), make_assume(cond, *pos));

  if (per_step_hints + direct_hints > 0)
    log_debug(
      "symmetry-breaking",
      "injected {} redundant extremum bound(s) ({} per-step, {} "
      "direct-to-final) from recognised max/min ite assignments",
      per_step_hints + direct_hints,
      per_step_hints,
      direct_hints);

  return true;
}
