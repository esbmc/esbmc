#include <esbmc/kind_invariants.h>
#include <esbmc/ranking_synthesis.h>
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <goto-programs/goto_houdini_invariants.h>
#include <goto-programs/goto_invariant_synthesis.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_utils.h>
#include <algorithm>
#include <set>
#include <string>

namespace
{
/// The share of kMaxCandidatesPerLoop the up-front sources may take, leaving
/// the rest for what counterexamples suggest later.
constexpr size_t kMaxGeneratedPerLoop = kMaxCandidatesPerLoop / 2;

/// Whether every jump into @p loop from outside lands on its head, where the
/// schema havocs. A switch or goto entering mid-body would run the body from a
/// concrete state and stop at the cut back edge, never reaching later
/// iterations (#7565).
bool entered_only_at_head(const loopst &loop, const goto_programt &body)
{
  const goto_programt::targett head = loop.get_original_loop_head();
  std::set<const goto_programt::instructiont *> inside;
  for (auto it = head;; ++it)
  {
    inside.insert(&*it);
    if (it == loop.get_original_loop_exit())
      break;
  }

  forall_goto_program_instructions (it, body)
  {
    if (!it->is_goto() || inside.count(&*it))
      continue;
    for (const auto &target : it->targets)
      if (target != head && inside.count(&*target))
        return false;
  }
  return true;
}

/// Whether the schema can cut @p loop without losing state. It havocs named
/// symbols, so a pointee written through a dereference -- in the loop or in
/// a callee -- would keep its concrete value across the abstract iteration and
/// let a guess that is not an invariant pass.
bool is_cuttable(const loopst &loop, goto_functiont &goto_function)
{
  return !loop.writes_through_pointer() && !loop.modifies_pointer_array() &&
         entered_only_at_head(loop, goto_function.body) &&
         !loop.get_modified_loop_vars().empty() &&
         !houdini_has_existing_invariant(
           loop.get_original_loop_head(),
           goto_function.body.instructions.begin());
}

/// Symbols, integer literals and the arithmetic over them: what a ranking
/// measure may contain to be meaningful at the loop head.
bool is_plain_arith(const expr2tc &e)
{
  if (is_symbol2t(e) || is_constant_int2t(e))
    return true;
  if (!is_typecast2t(e) && !is_add2t(e) && !is_sub2t(e) && !is_neg2t(e))
    return false;
  bool plain = true;
  e->foreach_operand(
    [&plain](const expr2tc &op) { plain = plain && is_plain_arith(op); });
  return plain;
}

/// `m >= L - 1` for each ranking measure (m, L) of the loop's guard: the
/// measure's value once the guard fails after a unit decrease.
std::vector<expr2tc> measure_floors(const loopst &loop)
{
  const goto_programt::targett head = loop.effective_loop_head();
  if (!head->is_goto())
    return {};

  std::vector<std::pair<expr2tc, expr2tc>> measures;
  measure_candidates_from_guard(loop_continuation(*head), measures);

  std::vector<expr2tc> floors;
  for (const auto &[m, L] : measures)
    if (is_plain_arith(m))
      floors.push_back(
        greaterthanequal2tc(m, sub2tc(m->type, L, gen_one(m->type))));
  return floors;
}

/// `continuation || P` for an assertion P right at the loop's exit: what every
/// head visit must satisfy for P to hold once the loop leaves.
std::vector<expr2tc> lifted_exit_assertions(const loopst &loop)
{
  const goto_programt::targett head = loop.effective_loop_head();
  if (!head->is_goto() || head->targets.size() != 1)
    return {};

  goto_programt::targett exit = head->targets.front();
  while (!exit->is_assert() && loop_invariant::is_inert_scan_instruction(exit))
    ++exit;
  if (!exit->is_assert() || reads_through_pointer(exit->guard))
    return {};

  return {or2tc(loop_continuation(*head), exit->guard)};
}

std::vector<expr2tc>
interval_bounds(const ait<interval_domaint> &intervals, const loopst &loop)
{
  const goto_programt::targett head = loop.effective_loop_head();
  if (!intervals.target_is_mapped(head))
    return {};

  std::vector<expr2tc> bounds;
  for (const expr2tc &var : loop.get_modified_loop_vars())
  {
    if (!is_symbol2t(var))
      continue;
    const expr2tc bound = intervals[head].make_expression(var);
    if (!is_true(bound))
      bounds.push_back(bound);
  }
  return bounds;
}
} // namespace

std::vector<kind_candidatet> generate_kind_candidates(
  goto_functionst &goto_functions,
  const optionst &options,
  const namespacet &ns,
  bool with_intervals)
{
  const overflow_checkst overflow{
    options.get_bool_option("overflow-check"),
    options.get_bool_option("unsigned-overflow-check")};

  interval_domaint::set_options(options);
  ait<interval_domaint> intervals;
  if (with_intervals)
    intervals(goto_functions, ns);

  std::vector<kind_candidatet> pool;
  Forall_goto_functions (f, goto_functions)
  {
    if (!f->second.body_available || f->second.body.hide)
      continue;

    goto_loopst loops(f->first, goto_functions, f->second);
    for (auto &loop : loops.get_loops())
    {
      const unsigned id = stamped_loop(*loop.get_original_loop_exit());
      if (!id || !is_cuttable(loop, f->second))
        continue;

      size_t added = 0;
      const size_t first = pool.size();
      auto add = [&](const std::vector<expr2tc> &exprs, const char *source) {
        for (const expr2tc &e : exprs)
        {
          if (added == kMaxGeneratedPerLoop)
            return;
          const bool seen = std::any_of(
            pool.begin() + first,
            pool.end(),
            [&e](const kind_candidatet &c) { return c.expr == e; });
          if (seen)
            continue;
          pool.push_back({id, e, source});
          ++added;
        }
      };

      add(affine_loop_invariants(f->second, loop, overflow), "affine");
      add(lifted_exit_assertions(loop), "property");
      add(measure_floors(loop), "ranking");
      add(interval_bounds(intervals, loop), "interval");
      add(houdini_template_candidates(loop, f->second.body), "template");
    }
  }
  return pool;
}

std::set<size_t> emit_kind_candidates(
  goto_functionst &goto_functions,
  const std::vector<kind_candidatet> &pool,
  const std::vector<size_t> &ids)
{
  std::set<size_t> emitted;
  std::map<unsigned, std::vector<size_t>> by_loop;
  for (size_t id : ids)
    by_loop[pool[id].loop].push_back(id);

  Forall_goto_functions (f, goto_functions)
  {
    if (!f->second.body_available || f->second.body.hide)
      continue;

    goto_loopst loops(f->first, goto_functions, f->second);
    for (auto &loop : loops.get_loops())
    {
      const unsigned loop_id = stamped_loop(*loop.get_original_loop_exit());
      if (!loop_id || !is_cuttable(loop, f->second))
        continue;

      const goto_programt::targett anchor = loop.get_original_loop_head();
      const auto found = by_loop.find(loop_id);
      if (found == by_loop.end())
      {
        houdini_emit_candidate(
          f->second, anchor, gen_true_expr(), "cut" + std::to_string(loop_id));
        continue;
      }
      for (size_t id : found->second)
      {
        houdini_emit_candidate(
          f->second, anchor, pool[id].expr, std::to_string(id));
        emitted.insert(id);
      }
    }
  }
  goto_functions.update();
  return emitted;
}

std::map<unsigned, goto_programt::targett>
insert_kind_placeholders(goto_functionst &goto_functions)
{
  std::map<unsigned, goto_programt::targett> placeholders;
  Forall_goto_functions (f, goto_functions)
  {
    Forall_goto_program_instructions (back_edge, f->second.body)
    {
      const unsigned id =
        back_edge->is_backwards_goto() ? stamped_loop(*back_edge) : 0;
      if (!id)
        continue;

      const goto_programt::targett at =
        skip_inductive_preamble(back_edge->targets.front(), back_edge);
      if (at == back_edge)
        continue;

      goto_programt::instructiont assume;
      assume.make_assumption(gen_true_expr());
      assume.inductive_step_instruction = true;
      assume.location = at->location;
      stamp_loop(assume, id);
      assume.function = at->function;
      f->second.body.insert_swap(at, assume);
      placeholders[id] = at;
    }
  }
  goto_functions.update();
  return placeholders;
}
