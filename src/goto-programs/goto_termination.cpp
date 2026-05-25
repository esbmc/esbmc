#include <goto-programs/goto_termination.h>
#include <goto-programs/goto_loops.h>
#include <util/expr_util.h>
#include <algorithm>
#include <map>
#include <vector>

void goto_termination(goto_functionst &goto_functions, optionst &options)
{
  goto_terminationt(goto_functions, options).run();
}

bool goto_terminationt::visit_function(
  const irep_idt &function_name,
  const goto_functiont &function) const
{
  if (!function.body_available)
    return false;
  // Skip __ESBMC_main: that's where the program-entry FUNCTION_CALL
  // sequence lives; running k-induction's havoc + entry-cond on it
  // would scan its (loop-free) body and potentially mutate it.
  if (function_name == "__ESBMC_main")
    return false;
  // Skip library helpers (converted with __ESBMC_HIDE; see
  // goto_convert_functions.cpp:39-51). Their per-loop markers would
  // be reachable from every program via stdlib/pthread chains and
  // defeat IS-UNSAT discrimination.
  if (function.body.hide)
    return false;
  return true;
}

void goto_terminationt::transform_loop(
  const irep_idt &function_name,
  goto_functiont &goto_function,
  loopst &loop)
{
  const auto &modified = loop.get_modified_loop_vars();

  if (!modified.empty())
  {
    kinduction.transform_loop(function_name, goto_function, loop);
    inject_noop_cycle_assumes(goto_function, loop);

    // Pointer-only-modified loops: k-induction's havoc emits
    // `ASSIGN p = NONDET(ptr_t)` for each modified pointer. Under
    // --add-symex-value-sets symex constrains the post-havoc value
    // to remain inside the pre-havoc points-to set, so the
    // dereferences are sound (they hit the same memory). But the
    // *contents* of that memory are still whatever the pre-loop
    // state placed there — the havoc doesn't randomise the
    // memory the pointers refer to. For loops whose termination
    // depends on what's read through those pointers (e.g.
    // strrchr's `while (*t != 0) ++t` walking a NONDET-length
    // string), IS sees a state where `*t != 0` for every k it
    // unwinds and reports non-termination spuriously.
    //
    // Pre-merge HEAD had a dedicated `has_pointer_only_loop`
    // function gating IS for exactly this shape; the upstream
    // k-induction merge removed it on the claim that the value-
    // set assume made it unnecessary. The assume protects pointer
    // *aliasing* but not pointed-to-data, so the gate is still
    // needed.
    bool all_pointer = true;
    for (const auto &v : modified)
    {
      if (!is_pointer_type(v->type))
      {
        all_pointer = false;
        break;
      }
    }
    if (all_pointer)
      any_unreliable_is_loop = true;
    return;
  }

  // The modified-var set is empty, so k-induction's
  // make_nondet_assign emits no havoc for this loop. Two distinct
  // shapes land here:
  //
  //   (a) The loop body is genuinely state-free — e.g. `while (1) {}`
  //       or `while (cond) ;` with `cond` and the body using only
  //       loop-invariant state. The marker is unreachable from any
  //       state; IS correctly proves non-termination without needing
  //       a havoc.
  //
  //   (b) The body writes through dereferences (`*p = ...`) or
  //       array indices. collect_lhs_symbols classifies the pointee
  //       as written but the symbol set tracks only the *pointer*
  //       (as read), so modified ends up empty. Without a havoc, IS
  //       sees only the concrete iterates 1..k of the loop and a
  //       marker-unreachable UNSAT within k unwindings is *not* a
  //       real non-termination witness — at higher k the loop may
  //       exit (e.g. termination9's binary counter).
  //
  // We can distinguish (a) from (b) by checking whether the body
  // contains any ASSIGN at all. (a) has none; (b) has at least one.
  // Only (b) is the soundness hazard; flag it so finalize() can
  // disable IS for the whole program.
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  for (auto p = loop_head; p != std::next(loop_exit); ++p)
  {
    if (p->is_assign())
    {
      any_unreliable_is_loop = true;
      break;
    }
  }
}

void goto_terminationt::after_function(
  const irep_idt &function_name,
  goto_functiont &goto_function)
{
  insert_markers_for_function(function_name, goto_function);
}

void goto_terminationt::finalize()
{
  if (any_unreliable_is_loop)
    options.set_option("disable-inductive-step", true);
}

void goto_terminationt::insert_markers_for_function(
  const irep_idt &function_name,
  goto_functiont &goto_function)
{
  // Per-loop termination markers: place an ASSERT(false) on every
  // edge that leaves a loop, preserving each edge's original
  // successor so CFG semantics are unchanged.
  //
  // Loop identification uses a fresh goto_loopst over the
  // post-transformation function body (the natural-loop notion is
  // stable under the havoc + entry-cond ASSUME inserts done by
  // k-induction). For each loop:
  //
  //   1. Walk the loop body and collect every forward GOTO whose
  //      target sits outside [loop_head, loop_exit] — these are the
  //      genuine exit edges (top-test, break, early-return-as-goto,
  //      goto-label-outside-loop, switch-case branches that escape).
  //   2. Group exits by their original target. For each unique
  //      target T, insert an ASSERT(false); GOTO T immediately
  //      before T, and retarget every exit edge that originally
  //      went to T to point at that marker. This preserves each
  //      edge's successor exactly.
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
  goto_programt &body = goto_function.body;
  goto_loopst loops(function_name, goto_functions, goto_function);
  if (loops.get_loops().empty())
    return;

  // Snapshot loop boundaries before mutating the body. Sort inner
  // loops first (back-edge location_number descending) so the
  // marker pass processes nearer loops first and outer-loop iterator
  // math stays stable.
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
      marker.function = function_name;
      marker.inductive_step_instruction = true;
      auto marker_it = body.instructions.insert(orig_target, marker);

      goto_programt::instructiont jump;
      jump.type = GOTO;
      jump.guard = gen_true_expr();
      jump.targets.push_back(orig_target);
      jump.location = orig_target->location;
      jump.function = function_name;
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
        marker.function = function_name;
        marker.inductive_step_instruction = true;
        body.instructions.insert(post, marker);
      }
    }
  }
}

void goto_terminationt::inject_noop_cycle_assumes(
  goto_functiont &goto_function,
  const loopst &loop)
{
  // Walk the loop body looking for IFs whose "guard taken" path
  // reaches the back-edge without crossing any state-modifying
  // instruction (ASSIGN or FUNCTION_CALL). The IF's guard is then a
  // no-op-cycle condition: when satisfied, the body has no effect
  // and the loop iterates forever.
  //
  // For each such IF, splice an `ASSUME(guard)` right before the
  // post-havoc loop_head IF and mark it inductive_step_instruction
  // so only IS sees it. IS then proves the marker unreachable from
  // any state satisfying entry-cond AND guard, which is a real
  // non-termination witness if such states exist.
  //
  // We run AFTER kinduction.transform_loop, so the loop_head IF has
  // moved to a new slot — `loop.get_original_loop_head()` still
  // refers to the original *iterator* but that slot now holds
  // kinduction's NONDET ASSIGN (instructiont::swap exchanges
  // contents but not location_number). The actual IF iterator is
  // wherever the back-edge points: kinduction's
  // adjust_loop_head_and_exit retargets the back-edge at loop_head
  // (or the entry-cond ASSUME just before it). The IF is therefore
  // reachable by walking the back-edge's target forward to the
  // first GOTO instruction.
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  if (loop_exit->targets.empty())
    return;
  goto_programt::targett post_havoc_head = *loop_exit->targets.begin();

  // The pre-havoc body range (used to detect no-op paths) is the
  // original [loop_head, loop_exit). We use the *original* iterators
  // here because they reference the unmodified body instructions
  // that survived kinduction's rewrite — insert_swap shuffled their
  // location_numbers but the iterators themselves still walk the
  // same physical sequence.
  goto_programt::targett orig_loop_head = loop.get_original_loop_head();
  if (orig_loop_head == loop_exit)
    return;

  // Find the actual IF by walking forward from orig_loop_head;
  // kinduction's mutation moved the IF's CONTENT past the inserted
  // NONDET/ASSUME instructions, so the original iterator now points
  // at the first NONDET. Walk forward to the first non-
  // inductive_step_instruction GOTO — that's the loop_head IF.
  goto_programt::targett if_it = orig_loop_head;
  while (if_it != loop_exit &&
         (if_it->inductive_step_instruction || !if_it->is_goto()))
    ++if_it;
  if (if_it == loop_exit)
    return;

  // Skip nested loops. An IF in an inner loop's body whose target
  // is "outside the inner loop" (the inner loop's exit) looks
  // structurally like a no-op-cycle branch of the *outer* loop —
  // from the outer's perspective the inner loop is "the body", and
  // the inner's exit-test guard would (incorrectly) be injected as
  // a no-op witness for the outer. Detect nesting by scanning for
  // any backwards-GOTO inside the body other than the loop's own
  // back-edge.
  for (auto q = std::next(if_it); q != loop_exit; ++q)
  {
    if (q->is_backwards_goto())
      return;
  }

  for (auto p = std::next(if_it); p != loop_exit; ++p)
  {
    if (!p->is_goto() || p->is_backwards_goto() || p->targets.size() != 1)
      continue;

    auto target = *p->targets.begin();
    // Only consider IFs whose taken branch stays *inside* the loop
    // body and skips over body instructions (toward the back-edge).
    // An IF whose target is outside [p, loop_exit] is an exit edge —
    // taking it leaves the loop, which is the opposite of a no-op
    // cycle (the loop *terminates* via that path).
    //
    // location_number comparisons aren't reliable after kinduction's
    // insert_swap (swap doesn't exchange location_number), so we
    // verify with an iterator walk: target must be reachable from
    // std::next(p) by walking forward without passing loop_exit.
    bool target_in_loop = false;
    for (auto q = std::next(p); q != std::next(loop_exit); ++q)
    {
      if (q == target)
      {
        target_in_loop = true;
        break;
      }
    }
    if (!target_in_loop)
      continue;

    // Walk from the IF's target to the back-edge via fall-through
    // (i.e., the path control will take if the IF is taken). If we
    // see any state-modifying instruction, this IF's taken path
    // doesn't preserve state — not a no-op cycle.
    bool noop = true;
    for (auto q = target; q != std::next(loop_exit); ++q)
    {
      if (q->is_assign() || q->is_function_call())
      {
        noop = false;
        break;
      }
    }
    if (!noop)
      continue;

    // p's guard is the IF's "taken" condition. Splice an ASSUME of
    // it right before post_havoc_head, inductive-step only. This
    // sits between kinduction's entry-cond ASSUME and the IF, so
    // it constrains the just-havoced state.
    goto_programt::instructiont assumption;
    assumption.type = ASSUME;
    assumption.guard = p->guard;
    assumption.location = post_havoc_head->location;
    assumption.location.comment("termination no-op-cycle witness");
    assumption.function = post_havoc_head->function;
    assumption.inductive_step_instruction = true;
    goto_function.body.instructions.insert(post_havoc_head, assumption);
  }
}
