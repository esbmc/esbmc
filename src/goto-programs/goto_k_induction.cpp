#include <goto-programs/goto_k_induction.h>
#include <goto-programs/remove_no_op.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>
#include <algorithm>
#include <map>

void goto_k_induction(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_k_inductiont(it->first, goto_functions, it->second);

  goto_functions.update();
}

void goto_termination(goto_functionst &goto_functions)
{
  // Termination as a safety property. Two transformations:
  //
  //   1. Apply the k-induction loop transformation (havoc loop vars +
  //      assume entry condition) to every loop. The inductive step
  //      then runs from an arbitrary loop state — so if it cannot
  //      reach end-of-main, no iterate of the loop can.
  //   2. Insert `assert(false)` right after the call to `main()` in
  //      `__ESBMC_main`. The reduction is:
  //
  //        program does NOT terminate  iff  assert is unreachable
  //
  //      In the inductive step, UNSAT (assert unreachable from havoc'd
  //      state) proves non-termination outright. In the base case,
  //      SAT (assert reachable from concrete initial state) refutes it.
  // Apply k-induction (havoc + assume_entry) to every function except
  // __ESBMC_main: that's where we insert the termination marker, and
  // running k-induction there would scan its (loop-free) body and
  // potentially mutate it.
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available && it->first != "__ESBMC_main")
      goto_k_inductiont(it->first, goto_functions, it->second);
  goto_functions.update();

  // Per-loop termination markers: place an ASSERT(false) on every
  // edge that leaves a loop, preserving each edge's original
  // successor so CFG semantics are unchanged.
  //
  // Loop identification uses goto_loopst (the same natural-loop
  // notion k-induction uses): for each loop we have an authoritative
  // loop_head and loop_exit (back-edge), and the loop's instruction
  // range is [loop_head, loop_exit].
  //
  // For each loop:
  //   1. Walk the loop body and collect every forward GOTO whose
  //      target sits outside [loop_head, loop_exit] — these are the
  //      genuine exit edges (top-test, break, early-return-as-goto,
  //      goto-label-outside-loop, switch-case branches that escape).
  //   2. Group exits by their original target. For each unique
  //      target T, insert an ASSERT(false); GOTO T immediately before
  //      T, and retarget every exit edge that originally went to T
  //      to point at that marker. This preserves each edge's
  //      successor exactly.
  //   3. For do-while loops (conditional back-edge), the natural
  //      fall-through past the back-edge is also an exit. Splice an
  //      ASSERT(false) at std::next(loop_exit); the marker falls
  //      through to its original successor and so preserves
  //      semantics for that path too.
  //
  // The retarget (rather than inserting at the original target) is
  // critical: exit-target labels are shared with outer branches that
  // converge there (e.g. `if (c == 0) goto L_after; while (...) goto
  // L_after;`). Inserting at the target would let the outer path
  // also hit the marker, defeating per-loop isolation.
  //
  // Marker is flagged inductive_step_instruction so BC/FC skip it;
  // only IS sees the claim.
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available || it->first == "__ESBMC_main")
      continue;
    // Only instrument user-program loops. Library functions are
    // converted with `body.hide = true` via the __ESBMC_HIDE label
    // (goto_convert_functions.cpp:39-51). Their per-loop markers
    // would be reachable from every program via stdlib/pthread
    // chains and defeat IS-UNSAT discrimination.
    if (it->second.body.hide)
      continue;

    goto_programt &body = it->second.body;
    goto_loopst loops(it->first, goto_functions, it->second);
    if (loops.get_loops().empty())
      continue;

    // Snapshot loop boundaries before mutating the body. Each loopst
    // stores iterators (head, back-edge); inserts invalidate nothing
    // for std::list, so iterators remain valid across the rewrites
    // below, but the *order* in which we process loops matters: we
    // process inner loops first (back-edge with smaller location
    // distance) so that markers for an inner loop are placed inside
    // an outer loop's body where appropriate. Sorting by back-edge
    // location number (descending) processes nearer (i.e. likely
    // inner) loops first.
    struct loop_info
    {
      goto_programt::targett head;
      goto_programt::targett back;
    };
    std::vector<loop_info> infos;
    for (const auto &loop : loops.get_loops())
      infos.push_back(
        {loop.get_original_loop_head(), loop.get_original_loop_exit()});
    std::sort(
      infos.begin(),
      infos.end(),
      [](const loop_info &a, const loop_info &b)
      { return a.back->location_number > b.back->location_number; });

    for (const auto &info : infos)
    {
      goto_programt::targett loop_head = info.head;
      goto_programt::targett loop_exit = info.back;
      const unsigned head_loc = loop_head->location_number;
      const unsigned exit_loc = loop_exit->location_number;

      // Collect every forward exit edge inside [loop_head, loop_exit].
      // An exit edge is a GOTO whose target lies outside the loop range
      // (either before loop_head or strictly past loop_exit).
      std::vector<goto_programt::targett> exit_edges;
      for (auto p = loop_head; p != std::next(loop_exit); ++p)
      {
        if (!p->is_goto() || p->is_backwards_goto())
          continue;
        if (p->targets.size() != 1)
          continue;
        auto tgt = *p->targets.begin();
        const unsigned tgt_loc = tgt->location_number;
        if (tgt_loc < head_loc || tgt_loc > exit_loc)
          exit_edges.push_back(p);
      }

      // Group exit edges by their original target so we emit one
      // marker per distinct successor — each marker falls through to
      // its own GOTO that restores the original target.
      std::map<unsigned, std::vector<goto_programt::targett>> by_target;
      std::map<unsigned, goto_programt::targett> target_iters;
      for (auto e : exit_edges)
      {
        auto tgt = *e->targets.begin();
        unsigned key = tgt->location_number;
        by_target[key].push_back(e);
        target_iters[key] = tgt;
      }

      for (auto &kv : by_target)
      {
        goto_programt::targett orig_target = target_iters[kv.first];
        if (orig_target == body.instructions.end())
          continue;

        // Insert ASSERT(false) + GOTO orig_target immediately before
        // orig_target. The new marker falls through to the new GOTO,
        // which jumps to orig_target — semantically equivalent to
        // jumping directly to orig_target for any caller that
        // reaches the marker.
        goto_programt::instructiont marker;
        marker.type = ASSERT;
        marker.guard = gen_false_expr();
        marker.location = orig_target->location;
        marker.location.comment("termination per-loop marker");
        marker.function = it->first;
        marker.inductive_step_instruction = true;
        auto marker_it = body.instructions.insert(orig_target, marker);

        goto_programt::instructiont jump;
        jump.type = GOTO;
        jump.guard = gen_true_expr();
        jump.targets.push_back(orig_target);
        jump.location = orig_target->location;
        jump.function = it->first;
        jump.inductive_step_instruction = true;
        body.instructions.insert(orig_target, jump);

        for (auto e : kv.second)
        {
          e->targets.clear();
          e->targets.push_back(marker_it);
        }
      }

      // Do-while fall-through: a conditional back-edge with a
      // non-trivially-true guard implies control falls through past
      // it on loop exit. Place an ASSERT(false) at std::next(loop_exit);
      // it falls through to its original successor so semantics are
      // preserved.
      if (!is_true(loop_exit->guard))
      {
        auto post = std::next(loop_exit);
        if (post != body.instructions.end())
        {
          goto_programt::instructiont marker;
          marker.type = ASSERT;
          marker.guard = gen_false_expr();
          marker.location = post->location;
          marker.location.comment("termination per-loop marker");
          marker.function = it->first;
          marker.inductive_step_instruction = true;
          body.instructions.insert(post, marker);
        }
      }
    }
  }
  goto_functions.update();

  // No global end-of-main marker: per-loop markers already capture
  // the only paths whose reachability matters for non-termination
  // (the loop's exit edge). A global marker would also fire on any
  // bypass path that skips the loop entirely (e.g. `if (c == 0)
  // while (...)` with c != 0), defeating IS-as-non-termination on
  // mixed-path programs. Loop-free programs hit FC UNSAT at k=1
  // (nothing to unwind) and report SUCCESSFUL via that route.
}

void goto_k_inductiont::goto_k_induction()
{
  // Full unwind the program
  for (auto &function_loop : function_loops)
  {
    if (function_loop.get_modified_loop_vars().empty())
      continue;

    // Start the loop conversion
    convert_finite_loop(function_loop);
  }
}

void goto_k_inductiont::convert_finite_loop(loopst &loop)
{
  // Get current loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // The branch cache is loop-scoped: a cached `reaches` value is computed
  // against this loop's `loop_exit`, and the cached guards were collected
  // along this loop's recursive walk. Nested loops share the goto_k_inductiont
  // instance (one per function), so without clearing here the outer loop's
  // entries would be reused by the inner loop with a different `loop_exit`.
  marked_branch.clear();

  guardst guards;
  get_entry_cond_rec(loop_head, loop_exit, guards);

  // Remove loop conditions not related to the written variables
  remove_unrelated_loop_cond(guards, loop);

  // Assume the loop entry condition before go into the loop
  assume_loop_entry_cond_before_loop(loop_head, loop_exit, guards);

  // Create the nondet assignments on the beginning of the loop
  make_nondet_assign(loop_head, loop);

  // Check if the loop exit needs to be updated
  // We must point to the assume that was inserted in the previous
  // transformation
  adjust_loop_head_and_exit(loop_head, loop_exit);
}

bool goto_k_inductiont::get_entry_cond_rec(
  const goto_programt::targett &loop_head,
  const goto_programt::targett &loop_exit,
  guardst &guards)
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
    auto it = marked_branch.find(tmp_head->location_number);
    if (it != marked_branch.end())
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

      // Get the branch number for caching
      auto const branch_number = tmp_head->location_number;

      // Walk the true branch
      bool true_branch = true;
      guardst true_branch_guard;
      if (!is_false(g))
      {
        true_branch_guard[branch_number].add(g);
        true_branch = get_entry_cond_rec(
          tmp_head->targets.front(), loop_exit, true_branch_guard);
      }

      // Walk the false branch
      bool false_branch = true;
      guardst false_branch_guard;
      if (!is_true(g))
      {
        goto_programt::targett new_tmp_head = tmp_head;
        make_not(g);
        false_branch_guard[branch_number].add(g);
        false_branch =
          get_entry_cond_rec(++new_tmp_head, loop_exit, false_branch_guard);
      }

      // Cache: store BOTH the recursion's reach-status at this branch
      // AND the guards that should be re-injected on a later cache hit.
      branch_cache_entryt entry;
      entry.reaches = false_branch && true_branch;

      // If both sides reach the end of the loop or if neither reaches it
      // we can ignore them
      if (!(false_branch ^ true_branch))
      {
        marked_branch[branch_number] = std::move(entry);
        return false_branch && true_branch;
      }

      // At least only one of the branches reach the end of the loop, so
      // collect the guards from the non-reaching side.
      if (!true_branch)
      {
        guards.insert(true_branch_guard.begin(), true_branch_guard.end());
        entry.guards_to_merge = std::move(true_branch_guard);
        marked_branch[branch_number] = std::move(entry);
        return false;
      }

      if (!false_branch)
      {
        guards.insert(false_branch_guard.begin(), false_branch_guard.end());
        entry.guards_to_merge = std::move(false_branch_guard);
        marked_branch[branch_number] = std::move(entry);
        return false;
      }
    }
  }

  return false;
}

void goto_k_inductiont::make_nondet_assign(
  goto_programt::targett &loop_head,
  const loopst &loop)
{
  // Track the original loop head
  auto const original_loop_head = loop_head;

  // Check if the loop_head is an assertion, and track it
  const bool is_assert = loop_head->is_assert();

  // If it's an assertion, adjust loop_head to insert assignments before it
  if ((is_assert) && loop_head != goto_function.body.instructions.begin())
  {
    --loop_head;
    // We add instructions before a GOTO instruction
    // So we ensure we have one here
    assert(loop_head->is_goto());
  }

  // Get the list of variables modified inside the loop
  auto const &loop_vars = loop.get_modified_loop_vars();

  goto_programt dest;
  for (auto const &lhs : loop_vars)
  {
    // Generate a nondeterministic value for the loop variable
    expr2tc rhs = gen_nondet(lhs->type);

    // Create an assignment instruction for the nondeterministic value
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->inductive_step_instruction = true;
    t->code = code_assign2tc(lhs, rhs);
    // Keep the same location as the loop head
    t->location = loop_head->location;
  }

  // Insert the generated assignments before the loop head in the program
  goto_function.body.insert_swap(loop_head, dest);

  // Get original head again
  // Since we are using insert_swap to keep the targets, the
  // original loop head as shifted to after the assume cond
  if (is_assert)
  {
    // Restore the original loop head if it was an assertion
    loop_head = original_loop_head;
    assert(loop_head->is_assert());
  }
  else
  {
    // Move past the inserted instructions during the inductive step
    while ((++loop_head)->inductive_step_instruction)
      ;
  }
}

static bool contains_rec(const expr2tc &expr, const loopst::loop_varst &vars)
{
  // Check this node first: if it's a tracked symbol, we're done.
  if (is_symbol2t(expr) && vars.find(expr) != vars.end())
    return true;

  // Otherwise recurse into operands and stop at the first match.
  bool res = false;
  expr->foreach_operand(
    [&vars, &res](const expr2tc &e)
    {
      if (res || is_nil_expr(e))
        return;
      res = contains_rec(e, vars);
    });
  return res;
}

void goto_k_inductiont::remove_unrelated_loop_cond(
  guardst &guards,
  loopst &loop)
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

void goto_k_inductiont::assume_loop_entry_cond_before_loop(
  goto_programt::targett &loop_head,
  goto_programt::targett &loop_exit,
  const guardst &guards)
{
  goto_programt::targett tmp_head = loop_head;
  for (; tmp_head != loop_exit; tmp_head++)
  {
    auto const g = guards.find(tmp_head->location_number);
    if (g == guards.end())
      continue;

    expr2tc loop_cond = g->second.as_expr();

    if (is_nil_expr(loop_cond))
      return;

    if (is_true(loop_cond) || is_false(loop_cond))
      return;

    goto_programt dest;
    assume_cond(loop_cond, dest, tmp_head->location);

    goto_function.body.insert_swap(tmp_head, dest);
  }
}

void goto_k_inductiont::adjust_loop_head_and_exit(
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

void goto_k_inductiont::assume_cond(
  const expr2tc &cond,
  goto_programt &dest,
  const locationt &loc)
{
  goto_programt tmp_e;
  goto_programt::targett e = tmp_e.add_instruction(ASSUME);
  e->inductive_step_instruction = true;
  e->guard = cond;
  e->location = loc;
  dest.destructive_append(tmp_e);
}
