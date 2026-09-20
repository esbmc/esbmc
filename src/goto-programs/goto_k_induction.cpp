#include <goto-programs/goto_k_induction.h>
#include <algorithm>
#include <functional>
#include <goto-programs/goto_loops.h>
#include <goto-programs/loopst.h>
#include <goto-programs/remove_no_op.h>
#include <irep2/irep2_expr.h>
#include <irep2/irep2_guard.h>
#include <pointer-analysis/andersen.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/irep/std_expr.h>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace
{
using instructiont = goto_programt::instructiont;
// Keyed on the branch itself, so a branch visited twice contributes one
// conjunct to the emitted ASSUME. location_number would not separate two
// branches a previous loop's transformation displaced: both carry 0.
using guardst = std::unordered_map<const instructiont *, guard2tc>;

/// Cached result of expanding a forward GOTO branch during the entry-
/// condition collection. The boolean is the recursion's return value at
/// this branch (`false_branch && true_branch`, i.e. true iff both
/// subbranches reach the loop end); the guardst is the set of guards
/// that should be merged into the caller's local guardst when this
/// cache entry fires. Storing only the boolean (the legacy design)
/// silently dropped these guards on every cache hit, weakening the
/// entry-condition assume.
struct branch_cache_entryt
{
  bool reaches;
  guardst guards_to_merge;
};
using marked_branchst =
  std::unordered_map<const instructiont *, branch_cache_entryt>;

/// Walk the loop body and collect, into @p guards, the conditions under
/// which control flows from @p loop_head to @p loop_exit. Used by
/// transform_loop to derive the entry condition of the loop (the
/// conjunction of every IF-branch guard taken along a body-reaching
/// path). @p cache is a loop-scoped memoisation table keyed on the
/// IF itself; transform_loop builds a fresh one per loop. Keying on
/// location_number instead collides on every instruction a previous
/// loop's transformation spliced in: those carry 0 until
/// goto_functionst::update() runs.
bool get_entry_cond_rec(
  const goto_programt::targett &loop_head,
  const goto_programt::targett &loop_exit,
  guardst &guards,
  marked_branchst &cache)
{
  // Let's walk the loop and collect the constraints to enter the
  // loop. This might be messy because of side-effects

  // entry and exit numbers
  auto const &entry_number = loop_head->location_number;
  auto const &exit_number = loop_exit->location_number;

  // We jumped outside the loop, don't collect this constraint
  if (entry_number > exit_number)
    return true;

  goto_programt::targett tmp_head = loop_head;
  for (; tmp_head != loop_exit; tmp_head++)
  {
    auto it = cache.find(&*tmp_head);
    if (it != cache.end())
    {
      // Re-inject the guards the first visit collected here. Storing
      // only `reaches` lost these guards on every cache hit and silently
      // weakened the entry-condition assume.
      guards.insert(
        it->second.guards_to_merge.begin(), it->second.guards_to_merge.end());
      return it->second.reaches;
    }

    /* TODO: disable this for now, it will be used for termination evaluation
     * in the future.

    // Return, assume(0) and assert(0) stop the execution, so ignore these
    // branches too
    if(tmp_head->is_return())
      return true;

    if(tmp_head->is_assume() || tmp_head->is_assert())
      if(is_false(tmp_head->guard))
        return true;
    */

    if (tmp_head->is_goto() && !tmp_head->is_backwards_goto())
    {
      expr2tc g = tmp_head->guard;
      simplify(g);

      // If the guard is false, we can skip it right away
      if (is_false(g))
        continue;

      // We need to walk the branches and collect constraints that force
      // the path inside the loop and reach the end of the loop body

      auto const branch = &*tmp_head;

      // Walk the true branch
      bool true_branch = true;
      guardst true_branch_guard;
      if (!is_false(g))
      {
        true_branch_guard[branch].add(g);
        true_branch = get_entry_cond_rec(
          tmp_head->targets.front(), loop_exit, true_branch_guard, cache);
      }

      // Walk the false branch
      bool false_branch = true;
      guardst false_branch_guard;
      if (!is_true(g))
      {
        goto_programt::targett new_tmp_head = tmp_head;
        make_not(g);
        false_branch_guard[branch].add(g);
        false_branch = get_entry_cond_rec(
          ++new_tmp_head, loop_exit, false_branch_guard, cache);
      }

      // Cache: store BOTH the recursion's reach-status at this branch
      // AND the guards that should be re-injected on a later cache hit.
      branch_cache_entryt entry;
      entry.reaches = false_branch && true_branch;

      // If both sides reach the end of the loop or if neither reaches it
      // we can ignore them
      if (!(false_branch ^ true_branch))
      {
        cache[branch] = std::move(entry);
        return false_branch && true_branch;
      }

      // At least only one of the branches reach the end of the loop, so
      // collect the guards from the non-reaching side.
      if (!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        entry.guards_to_merge = std::move(true_branch_guard);
        cache[branch] = std::move(entry);
        return false;
      }

      if (!false_branch)
      {
        guards.insert(false_branch_guard.begin(), false_branch_guard.end());
        entry.guards_to_merge = std::move(false_branch_guard);
        cache[branch] = std::move(entry);
        return false;
      }
    }
  }

  return false;
}

/// The slot insert_swap writes the havoc block to. Every edge that reaches
/// this instruction crosses the block — insert_swap keeps jumps pinned to
/// the iterator and leaves the original content in the block's fall-through
/// path — and no other edge does.
goto_programt::targett
havoc_slot(goto_functiont &goto_function, goto_programt::targett loop_head)
{
  if (
    loop_head->is_assert() &&
    loop_head != goto_function.body.instructions.begin())
  {
    --loop_head;
    // We add instructions before a GOTO instruction
    // So we ensure we have one here
    assert(loop_head->is_goto());
  }
  return loop_head;
}

/// loop_varst is an unordered_set hashed by irep2_hash, which folds in
/// irep_idt::hash() -- the string's interning sequence number, not its text.
/// Iteration order therefore depends on what else has been interned earlier in
/// the run, so an unrelated pass that interns a string permutes these havocs
/// (docs/roadmap/scope-clang-c-irep2.md §13). Order by printed form, which is
/// stable across runs; the assignments are mutually independent, so only the
/// order changes.
std::vector<expr2tc> ordered_modified_vars(const loopst &loop)
{
  auto const &loop_vars = loop.get_modified_loop_vars();
  std::vector<expr2tc> ordered(loop_vars.begin(), loop_vars.end());
  std::sort(
    ordered.begin(), ordered.end(), [](const expr2tc &a, const expr2tc &b) {
      return a->pretty() < b->pretty();
    });
  return ordered;
}

void add_havoc_assigns(
  goto_programt &dest,
  const std::vector<expr2tc> &vars,
  const locationt &location)
{
  for (auto const &lhs : vars)
  {
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->inductive_step_instruction = true;
    t->code = code_assign2tc(lhs, gen_nondet(lhs->type));
    t->location = location;
  }
}

void make_nondet_assign(
  goto_functiont &goto_function,
  goto_programt::targett &loop_head,
  goto_programt::targett slot,
  const std::vector<expr2tc> &vars)
{
  const goto_programt::targett original_loop_head = loop_head;
  loop_head = slot;

  goto_programt dest;
  add_havoc_assigns(dest, vars, loop_head->location);
  goto_function.body.insert_swap(loop_head, dest);

  // insert_swap leaves loop_head on the first inserted instruction, so put it
  // back on the original head: exactly vars.size() forward, and never the old
  // "walk while inductive_step_instruction" heuristic, which also swallowed an
  // ASSUME a previous pass had left after the head and so retargeted the back
  // edge past the loop's exit IF.
  if (slot != original_loop_head)
    loop_head = original_loop_head;
  else
    std::advance(loop_head, vars.size());
}

bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
{
  // Check this node first: if it's a tracked symbol or pointee, we're done.
  if (vars.find(expr) != vars.end())
    return true;

  // Otherwise recurse into operands and stop at the first match.
  bool res = false;
  expr->foreach_operand([&vars, &res](const expr2tc &e) {
    if (res || is_nil_expr(e))
      return;
    res = contains_rec(e, vars);
  });
  return res;
}

void remove_unrelated_loop_cond(guardst &guards, const loopst &loop)
{
  auto const &loop_vars = loop.get_modified_loop_vars();
  if (!loop_vars.size())
  {
    guards.clear();
    return;
  }

  guardst::iterator g = guards.begin();
  while (g != guards.end())
  {
    expr2tc g_expr = g->second.as_expr();

    if (!contains_rec(g_expr, loop_vars))
      g = guards.erase(g);
    else
      ++g;
  }
}

void assume_loop_entry_cond_before_loop(
  goto_functiont &goto_function,
  goto_programt::targett &loop_head,
  const guardst &guards)
{
  // Combine all per-branch loop-entry guards into one ASSUME and
  // insert it via raw `insert(loop_head, ...)` immediately before the
  // loop_head IF. This runs *after* make_nondet_assign in
  // transform_loop, so the layout becomes
  //
  //   NONDET havocs               [inductive_step]   <- make_nondet_assign
  //   ASSUME(entry_cond)          [inductive_step]   <- this insert
  //   loop_head: IF !cond GOTO exit                  <- original IF
  //   body
  //   GOTO loop_head  -> back-edge targets the IF
  //   exit:
  //
  // Two properties matter for soundness/precision:
  //   1. The ASSUME comes *after* the NONDET havocs, so it constrains
  //      the just-havoced state (the actual inductive-hypothesis pin),
  //      not the pre-loop concrete values.
  //   2. The ASSUME sits *before* loop_head and is reached only via
  //      fall-through from the havocs. The back-edge
  //      (adjust_loop_head_and_exit retargets it at loop_head, the
  //      IF) skips both the havocs and the ASSUME on every iteration
  //      after the first, so the natural exit on `cond` is never
  //      blocked by a re-firing entry-cond assume.
  //
  // The legacy `insert_swap(tmp_head, ASSUME)` placement (one ASSUME
  // per branch, swapped *at* the branch instruction) was unsound
  // for the loop_head IF because insert_swap pins external jumps to
  // the iterator: back-edges then landed on the ASSUME and re-fired
  // it on every iteration, killing the natural-exit path in IS and
  // producing vacuous UNSAT proofs (e.g. SV-COMP
  // sll_of_sll_nondet_append-2). The earlier fix that combined the
  // ASSUME and placed it *after* loop_head fixed SLL but over-
  // constrained loops where the body modifies a loop-exit variable
  // (e.g. `i++` in array_3-1): the back-edge re-evaluated the
  // ASSUME on the iteration where the body's increment had pushed
  // the variable past the exit condition, again killing a natural
  // path.
  //
  // Iterate the collected guards directly instead of walking the
  // instruction range and looking each one up: after
  // make_nondet_assign's insert_swap the walk-and-lookup approach
  // missed every guard, so the combined entry condition came out empty
  // and no ASSUME was inserted, regressing post-loop assertions in
  // loop-invariants, incremental-smt, witnesses_validate and
  // esbmc-solidity (issue #4846).
  guard2tc combined;
  for (auto const &kv : guards)
  {
    expr2tc loop_cond = kv.second.as_expr();

    if (is_nil_expr(loop_cond) || is_true(loop_cond))
      continue;

    // A guard that simplifies to false would make the assume kill the
    // path even before the loop. Preserve the legacy "bail out" choice:
    // any unresolved false among the branch guards skips the whole
    // entry-cond instrumentation.
    if (is_false(loop_cond))
      return;

    combined.add(loop_cond);
  }

  expr2tc combined_expr = combined.as_expr();
  if (
    is_nil_expr(combined_expr) || is_true(combined_expr) ||
    is_false(combined_expr))
    return;

  goto_programt::instructiont instruction;
  instruction.type = ASSUME;
  instruction.guard = combined_expr;
  instruction.inductive_step_instruction = true;
  instruction.location = loop_head->location;
  goto_function.body.instructions.insert(loop_head, instruction);
}

void adjust_loop_head_and_exit(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit)
{
  loop_exit->targets.clear();
  loop_exit->targets.push_front(loop_head);

  goto_programt::targett _loop_exit = loop_exit;
  ++_loop_exit;

  // Zero means that the instruction was added during
  // the k-induction transformation
  if (_loop_exit->location_number == 0)
  {
    // Clear the target
    loop_head->targets.clear();

    // And set the target to be the newly inserted assume(cond)
    loop_head->targets.push_front(_loop_exit);
  }
}

/// Whether control reaches the loop's back edge from @p from without leaving
/// @p loop_range. The program-order range between the head and the back edge
/// over-approximates the loop body: a block that only jumps out of the loop
/// can sit inside the range, and havocing an edge into that block would
/// clobber the loop's variables on a path that never runs the loop.
bool reaches_back_edge(
  const goto_programt &body,
  goto_programt::const_targett from,
  const std::unordered_set<const instructiont *> &loop_range,
  const instructiont *back_edge)
{
  std::unordered_set<const instructiont *> seen;
  std::vector<goto_programt::const_targett> work{from};
  while (!work.empty())
  {
    const goto_programt::const_targett it = work.back();
    work.pop_back();
    if (&*it == back_edge)
      return true;
    if (!loop_range.count(&*it) || !seen.insert(&*it).second)
      continue;

    goto_programt::const_targetst successors;
    body.get_successors(it, successors);
    work.insert(work.end(), successors.begin(), successors.end());
  }
  return false;
}

/// The unconditional jumps that enter the loop somewhere the legacy havoc
/// placement does not cover: a rotated (bottom-test) loop is entered at its
/// test rather than at its head, and a switch dispatches through trampolines
/// that jump past the head into the middle of the body. Without a havoc on
/// those edges the inductive step symexes the concrete initial state and
/// truncates at k instead of inducting (#7565). @p slot itself, and every jump
/// to it, is already covered: insert_swap pins those to the block it writes
/// there.
///
/// Only unconditional jumps qualify. The havocs go in front of the jump
/// itself, so every execution reaching them goes on to enter the loop; a
/// conditional jump also falls through to a non-entry path, so splitting that
/// edge would need a trampoline block, and the new edges would have to be
/// revalidated against goto_loopst, adjust_loop_head_and_exit's
/// location_number == 0 heuristic and remove_unreachable. A loop whose only
/// entry is conditional therefore stays un-havoced and can still be falsely
/// proved -- one #7565 residual, pinned by
/// regression/k-induction/github_7565_conditional_entry_fail.
std::vector<goto_programt::targett> collect_entry_jumps(
  goto_programt &body,
  const loopst &loop,
  goto_programt::const_targett slot)
{
  std::unordered_set<const instructiont *> loop_range;
  for (goto_programt::targett it = loop.get_original_loop_head();
       it != body.instructions.end();
       ++it)
  {
    loop_range.insert(&*it);
    if (it == loop.get_original_loop_exit())
      break;
  }
  const instructiont *back_edge = &*loop.get_original_loop_exit();

  std::vector<goto_programt::targett> jumps;
  for (auto it = body.instructions.begin(); it != body.instructions.end(); ++it)
  {
    if (
      loop_range.count(&*it) || it == slot || !it->is_goto() ||
      !is_true(it->guard))
      continue;
    for (const auto &target : it->targets)
      if (
        target != slot && loop_range.count(&*target) &&
        reaches_back_edge(body, target, loop_range, back_edge))
      {
        jumps.push_back(it);
        break;
      }
  }
  return jumps;
}

/// Havoc the loop's modified variables on each edge in @p jumps, by splicing
/// the assignments in front of the jump. insert_swap keeps them on every
/// incoming edge, and because nothing but assignments is added the control
/// flow graph is unchanged -- no block to place, no edge for loop detection
/// to mistake for a back edge, and nothing for remove_unreachable to strip.
/// Every other edge into the jump crosses the havocs too, which on an
/// irreducible CFG costs precision but never soundness: a nondet assignment
/// subsumes the concrete one it replaces.
void havoc_entry_jumps(
  goto_programt &body,
  const std::vector<expr2tc> &vars,
  const std::vector<goto_programt::targett> &jumps)
{
  for (const goto_programt::targett &jump : jumps)
  {
    // Only the slot is insert_swapped between collection and here, and
    // collect_entry_jumps excluded it, so the iterator still holds the GOTO.
    assert(jump->is_goto() && is_true(jump->guard));
    goto_programt havocs;
    add_havoc_assigns(havocs, vars, jump->location);
    body.insert_swap(jump, havocs);
  }
}

/// Per-loop k-induction transformation: havoc each loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head.
void transform_loop(goto_functiont &goto_function, loopst &loop)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  goto_programt::targett const slot = havoc_slot(goto_function, loop_head);

  // Collected here, applied last: splicing a havoc block displaces the GOTO
  // onto a fresh instruction whose location_number is 0, which is what
  // adjust_loop_head_and_exit keys its loop-exit test on.
  const std::vector<goto_programt::targett> entry_jumps =
    collect_entry_jumps(goto_function.body, loop, slot);
  const std::vector<expr2tc> vars = ordered_modified_vars(loop);

  // Loop-scoped cache for get_entry_cond_rec. Nested loops in the same
  // function don't reuse entries because we construct a fresh cache per
  // call.
  marked_branchst cache;
  guardst guards;
  get_entry_cond_rec(loop_head, loop_exit, guards, cache);

  // Remove loop conditions not related to the written variables
  remove_unrelated_loop_cond(guards, loop);

  // Order matters: the entry-cond ASSUME must constrain the state
  // *after* the loop vars have been havoced, so we emit the NONDET
  // havocs first and then place the ASSUME just before loop_head.
  // make_nondet_assign also rewinds loop_head back onto the original
  // loop head (post its advance-by-`inserted` step), so by the time
  // assume_loop_entry_cond_before_loop runs the iterator is correct
  // and the ASSUME ends up between the havocs and the IF.

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(goto_function, loop_head, slot, vars);

  // Assume the loop entry condition before going into the loop
  assume_loop_entry_cond_before_loop(goto_function, loop_head, guards);

  // Check if the loop exit needs to be updated. We must point to the
  // assume that was inserted in the previous transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);

  havoc_entry_jumps(goto_function.body, vars, entry_jumps);
}

/// What a pointer may point to: the named objects, and whether it may also
/// reach the heap or anything at all.
struct targetst
{
  loopst::loop_varst named;
  bool heap = false;
  bool anything = false;
};

targetst
targets_of(andersent &points_to, const loopst &loop, const expr2tc &ptr)
{
  value_setst::valuest values;
  points_to.get_values(loop.get_original_loop_head(), ptr, values);
  targetst t;
  for (const expr2tc &v : values)
  {
    const expr2tc object =
      is_object_descriptor2t(v) ? to_object_descriptor2t(v).object : expr2tc();
    if (!is_nil_expr(object) && is_symbol2t(object) && check_var_name(object))
      t.named.insert(object);
    else if (!is_nil_expr(object) && is_dynamic_object2t(object))
      t.heap = true;
    else
      t.anything = true;
  }
  return t;
}

/// Adds the objects \p ptr may point to to \p objects, or returns false when
/// one of them has no name to havoc.
bool named_targets(
  andersent &points_to,
  const loopst &loop,
  const expr2tc &ptr,
  loopst::loop_varst &objects)
{
  // A pointer with no target (only ever nondet or null) writes no named
  // object: symex sends such a write to an invalid object.
  const targetst t = targets_of(points_to, loop, ptr);
  if (t.heap || t.anything)
  {
    if (messaget::state.target("k-induction", VerbosityLevel::Debug))
      log_debug(
        "k-induction",
        "cannot name what {} may point to at {}: {}",
        ptr->pretty(0),
        loop.get_original_loop_head()->location.as_string(),
        t.anything ? "anything" : "heap");
    return false;
  }
  objects.insert(t.named.begin(), t.named.end());
  return true;
}

bool names(const loopst::loop_varst &vars, const irep_idt &name)
{
  return std::any_of(vars.begin(), vars.end(), [&name](const expr2tc &v) {
    return is_symbol2t(v) && to_symbol2t(v).thename == name;
  });
}

/// The inductive step havocs only its loop's modified variables, so storage
/// the loop writes through a pointer would keep its pre-loop value and the
/// step would prove too much (#5224). Add that storage to the modified
/// variables: `*p` itself for a write inside `*p` while the loop leaves `p`
/// alone, otherwise the named objects the whole-program points-to sets
/// resolve the written pointer to. Returns false, and the caller disables the
/// inductive step, when neither covers a write.
bool havoc_written_objects(
  loopst &loop,
  andersent &points_to,
  const std::unordered_set<irep_idt, irep_id_hash> &address_taken)
{
  if (loop.unnamed_pointer_write())
    return false;

  loopst::loop_varst objects;
  for (const expr2tc &ptr : loop.get_written_pointers())
    if (!named_targets(points_to, loop, ptr, objects))
      return false;

  // A write through one pointee may move another's pointer (`*pp = r` moves
  // p). A pointer reaching anything can only move a pointer whose address
  // is taken somewhere.
  loopst::loop_varst clobbered;
  bool clobbers_anything = false;
  for (const expr2tc &pointee : loop.get_written_pointees())
  {
    const targetst t =
      targets_of(points_to, loop, to_dereference2t(pointee).value);
    clobbered.insert(t.named.begin(), t.named.end());
    clobbers_anything |= t.anything;
  }

  // check_var_name also filters the modified set, so a pointer it rejects may
  // be reassigned unseen.
  std::vector<expr2tc> pointees(
    loop.get_written_pointees().begin(), loop.get_written_pointees().end());
  const auto moves = [&](const expr2tc &pointee) {
    const expr2tc &ptr = to_dereference2t(pointee).value;
    const irep_idt &name = to_symbol2t(ptr).thename;
    return !check_var_name(ptr) || names(loop.get_modified_loop_vars(), name) ||
           names(objects, name) || names(clobbered, name) ||
           (clobbers_anything && address_taken.count(name));
  };
  for (auto it = std::find_if(pointees.begin(), pointees.end(), moves);
       it != pointees.end();
       it = std::find_if(pointees.begin(), pointees.end(), moves))
  {
    if (!named_targets(points_to, loop, to_dereference2t(*it).value, objects))
      return false;
    pointees.erase(it);
  }

  for (const expr2tc &obj : objects)
    loop.add_modified_var_to_loop(obj);
  for (const expr2tc &pointee : pointees)
    loop.add_modified_var_to_loop(pointee);
  return true;
}

void for_each_subexpr(
  const expr2tc &e,
  const std::function<void(const expr2tc &)> &f)
{
  if (is_nil_expr(e))
    return;
  f(e);
  e->foreach_operand([&f](const expr2tc &op) { for_each_subexpr(op, f); });
}

/// The functions the entry point may reach, calls through function pointers
/// included: every function named in a reachable body.
std::unordered_set<irep_idt, irep_id_hash>
reachable_functions(const goto_functionst &goto_functions)
{
  std::unordered_set<irep_idt, irep_id_hash> seen{goto_functions.main_id()};
  std::vector<irep_idt> work{goto_functions.main_id()};
  while (!work.empty())
  {
    auto it = goto_functions.function_map.find(work.back());
    work.pop_back();
    if (it == goto_functions.function_map.end() || !it->second.body_available)
      continue;
    for (const auto &instr : it->second.body.instructions)
      for (const expr2tc &e : {instr.code, instr.guard})
        for_each_subexpr(e, [&](const expr2tc &sub) {
          if (
            is_symbol2t(sub) && is_code_type(sub->type) &&
            seen.insert(to_symbol2t(sub).thename).second)
            work.push_back(to_symbol2t(sub).thename);
        });
  }
  return seen;
}

/// The variables whose address the program takes anywhere.
std::unordered_set<irep_idt, irep_id_hash>
address_taken_symbols(const goto_functionst &goto_functions)
{
  std::unordered_set<irep_idt, irep_id_hash> taken;
  forall_goto_functions (it, goto_functions)
    for (const auto &instr : it->second.body.instructions)
      for (const expr2tc &e : {instr.code, instr.guard})
        for_each_subexpr(e, [&taken](const expr2tc &sub) {
          if (!is_address_of2t(sub))
            return;
          for_each_subexpr(to_address_of2t(sub).ptr_obj, [&](const expr2tc &o) {
            if (is_symbol2t(o))
              taken.insert(to_symbol2t(o).thename);
          });
        });
  return taken;
}

/// True iff the program contains a reachable call to __VERIFIER_nondet_memory
/// in user (non-hidden) code. That intrinsic havocs a caller object with fresh
/// nondeterminism through a pointer (`*(p + i) = nondet_uchar()` over memory
/// with no nameable symbol) — the opaque "havoc_memory" analogue of the
/// per-element user nondet-array builder loops the SV-COMP "havoc_object" shape
/// uses. The inductive step cannot generalise such an input, so a reachable
/// call makes it unsound: it proves unsafe programs SAFE (the
/// SoftwareSystems-Intel-TDX-Module-ReachSafety `*_havoc_memory` tasks). The
/// model body is linked into *every* program, so gate on an actual call, not
/// on its always-present loop. See #5593 (recurrence of #5224 / #5230).
bool calls_nondet_memory(const goto_functionst &goto_functions)
{
  forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->second.body.hide)
      continue;
    for (const auto &instr : it->second.body.instructions)
    {
      if (!instr.is_function_call())
        continue;
      const code_function_call2t &call = to_code_function_call2t(instr.code);
      if (
        is_symbol2t(call.function) &&
        to_symbol2t(call.function).thename == "c:@F@__VERIFIER_nondet_memory")
        return true;
    }
  }
  return false;
}
} // namespace

bool goto_k_induction(goto_functionst &goto_functions, const namespacet &)
{
  // Build the points-to sets once, up front, on the pristine program: the
  // havoc a transformed loop gains would widen every later query to TOP.
  andersent points_to;
  points_to(goto_functions);
  const auto reachable = reachable_functions(goto_functions);
  const auto address_taken = address_taken_symbols(goto_functions);

  // A reachable __VERIFIER_nondet_memory call havocs a caller object the
  // inductive step cannot generalise, so its unsoundness is independent of any
  // loop shape — disable the inductive step up front. See #5593.
  bool disable_inductive_step = calls_nondet_memory(goto_functions);
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    // Library helpers (body.hide) write through pointers in nearly every
    // model (memcpy, string ops, ...). Letting them set the gate would
    // disable the inductive step for essentially every program. Their loops
    // are never the user property's witness, so exclude them from the
    // decision — mirrors the body.hide guard in goto_termination. So is a
    // function nothing calls.
    const bool decides =
      !it->second.body.hide && reachable.count(it->first) != 0;
    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      // Before the empty-modified-set skip: a loop that only writes through
      // pointers has no named modified variables until they are resolved.
      if (
        loop.writes_through_pointer() &&
        !havoc_written_objects(loop, points_to, address_taken) && decides)
        disable_inductive_step = true;

      if (loop.get_modified_loop_vars().empty())
        continue;
      transform_loop(it->second, loop);
    }
  }
  goto_functions.update();
  return disable_inductive_step;
}
