#include <goto-programs/goto_termination.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loops.h>
#include <util/expr_util.h>
#include <algorithm>
#include <map>
#include <vector>

void goto_termination(goto_functionst &goto_functions)
{
  // Termination as a safety property. Two transformations:
  //
  //   1. Apply the k-induction loop transformation (havoc loop vars +
  //      assume entry condition) to every loop. The inductive step
  //      then runs from an arbitrary loop state — so if it cannot
  //      reach end-of-main, no iterate of the loop can.
  //   2. Insert `assert(false)` per loop-exit edge. The reduction is:
  //
  //        program does NOT terminate  iff  every marker is unreachable
  //
  //      In the inductive step, UNSAT (marker unreachable from havoc'd
  //      state) proves non-termination outright. In the base case,
  //      SAT (marker reachable from concrete initial state) refutes it.
  //
  // Apply k-induction (havoc + assume_entry) to every function except
  // __ESBMC_main: that's where we insert the termination marker, and
  // running k-induction there would scan its (loop-free) body and
  // potentially mutate it.
  //
  // We can't just call goto_k_induction(goto_functions) because the
  // base run() loop doesn't skip __ESBMC_main. Instead, instantiate
  // once and drive transform_loop manually per function/loop.
  // Commit C will fold this into a proper subclass.
  {
    goto_k_inductiont kind(goto_functions);
    Forall_goto_functions (it, goto_functions)
    {
      if (!it->second.body_available || it->first == "__ESBMC_main")
        continue;
      goto_loopst loops(it->first, goto_functions, it->second);
      for (auto &loop : loops.get_loops())
      {
        if (loop.get_modified_loop_vars().empty())
          continue;
        kind.transform_loop(it->first, it->second, loop);
      }
    }
  }
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
