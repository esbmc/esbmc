#include <goto-programs/goto_termination.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loops.h>
#include <irep2/irep2_expr.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/std_expr.h>
#include <algorithm>
#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

namespace
{
/// Look for "no-op-cycle" branches in the loop body: an IF whose
/// guard-taken path reaches the back-edge without crossing any
/// state-modifying instruction. If found, inject ASSUME(guard) as
/// an inductive-step-only instruction right before the loop_head,
/// pinning IS to a fixed-point state.
///
/// Catches programs like Ex03 (`while (i < 0) if (i != -5) i++;`)
/// where there is a state (`i = -5`) at which the loop body has no
/// effect, so the loop is non-terminating. K-induction's vanilla IS
/// can't decide these — for an arbitrary havoced state, most
/// iterates do reach the marker, so IS reports SAT ("unable to
/// prove"). Pinning to the fixed-point state lets IS prove
/// marker-unreachable → UNSAT → non-termination.
///
/// Multiple no-op paths are conjoined: each contributes an
/// independent ASSUME. The first one whose guard is satisfiable
/// (together with the entry condition) yields the witness.
///
/// Runs AFTER goto_k_induction has transformed every loop, so the
/// loop_head IF has moved (insert_swap shuffles instruction content
/// between slots without exchanging location_number). The actual IF
/// iterator is found via the back-edge's target — kinduction's
/// adjust_loop_head_and_exit retargets the back-edge at the post-
/// havoc loop_head.
void inject_noop_cycle_assumes(
  goto_functiont &goto_function,
  const loopst &loop)
{
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

  // Skip nested loops. An IF in an inner loop's body whose target is
  // "outside the inner loop" (the inner loop's exit) looks
  // structurally like a no-op-cycle branch of the *outer* loop —
  // from the outer's perspective the inner loop is "the body", and
  // the inner's exit-test guard would (incorrectly) be injected as a
  // no-op witness for the outer. Detect nesting by scanning for any
  // backwards-GOTO inside the body other than the loop's own back-
  // edge.
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

    // Walk from the IF's target toward the back-edge, following
    // control flow. A genuine no-op cycle's taken path must reach the
    // back-edge (loop_exit) while preserving state. Two ways the path
    // can fail to be a no-op cycle:
    //
    //   1. It crosses a state-modifying instruction (ASSIGN / call) —
    //      the cycle changes state, so it isn't a fixed point.
    //
    //   2. It takes an unconditional forward GOTO whose target lies
    //      *past* the back-edge — that's an exit edge (e.g. a `break`
    //      lowered to `GOTO loop_end`). Taking it leaves the loop, the
    //      opposite of a no-op cycle. The earlier target_in_loop check
    //      only confirms the IF's *immediate* target is physically in
    //      range; it doesn't follow a subsequent jump out, which is
    //      exactly the soft_float `if (m >= cap) break;` shape that
    //      otherwise gets a contradictory ASSUME(m >= cap) injected
    //      against the entry condition m < cap (vacuous IS-UNSAT,
    //      spurious non-termination).
    bool noop = true;
    for (auto q = target; q != std::next(loop_exit); ++q)
    {
      if (q->is_assign() || q->is_function_call())
      {
        noop = false;
        break;
      }
      if (q->is_goto() && !q->is_backwards_goto() && q->targets.size() == 1)
      {
        // Does this forward GOTO jump past the back-edge (i.e. out of
        // the loop)? Scan [std::next(q), loop_exit]: if the target is
        // not found in that range, it lies beyond loop_exit — an exit.
        auto gt = *q->targets.begin();
        bool stays_in_loop = false;
        for (auto r = std::next(q); r != std::next(loop_exit); ++r)
        {
          if (r == gt)
          {
            stays_in_loop = true;
            break;
          }
        }
        if (!stays_in_loop)
        {
          noop = false;
          break;
        }
      }
    }
    if (!noop)
      continue;

    // The witness ASSUME(p->guard) is only sound if p->guard is a
    // genuine fixed-point invariant of the cycle — i.e. its value is
    // the same on entry to every iteration. If any variable in the
    // guard is *reassigned* earlier in the same iteration (between the
    // loop head and the IF), then the guard is recomputed each round
    // and pinning it does not describe a real recurrent state. The
    // canonical offender is `while (1) { x = nondet(); if (x == 0)
    // return; }`: the `if (x != 0)` continue-branch looks like a
    // no-op cycle, but x is freshly havoced every iteration, so
    // ASSUME(x != 0) just pins the loop to the non-exiting branch and
    // manufactures non-termination. Skip the witness in that case.
    std::unordered_set<irep_idt, irep_id_hash> guard_syms;
    {
      std::function<void(const expr2tc &)> collect = [&](const expr2tc &e) {
        if (is_nil_expr(e))
          return;
        if (is_symbol2t(e))
          guard_syms.insert(to_symbol2t(e).thename);
        else
          e->foreach_operand([&](const expr2tc &c) { collect(c); });
      };
      collect(p->guard);
    }
    bool guard_var_reassigned = false;
    for (auto q = if_it; q != p && !guard_var_reassigned; ++q)
    {
      if (q->is_assign())
      {
        const code_assign2t &a = to_code_assign2t(q->code);
        if (
          is_symbol2t(a.target) &&
          guard_syms.count(to_symbol2t(a.target).thename))
          guard_var_reassigned = true;
      }
      else if (q->is_function_call())
      {
        const code_function_call2t &fc = to_code_function_call2t(q->code);
        if (
          !is_nil_expr(fc.ret) && is_symbol2t(fc.ret) &&
          guard_syms.count(to_symbol2t(fc.ret).thename))
          guard_var_reassigned = true;
      }
    }
    if (guard_var_reassigned)
      continue;

    // p's guard is the IF's "taken" condition. Splice an ASSUME of
    // it right before post_havoc_head, inductive-step only. This
    // sits between kinduction's entry-cond ASSUME and the IF, so it
    // constrains the just-havoced state.
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

/// Decide whether this loop makes IS UNSAT unsound as a non-
/// termination witness. Returns true to indicate the soundness
/// hazard; the caller flips disable-inductive-step for the whole
/// program. See the two shape descriptions in the body.
bool loop_is_is_unreliable(const loopst &loop)
{
  const auto &modified = loop.get_modified_loop_vars();

  if (!modified.empty())
  {
    // Pointer-only-modified loops: k-induction's havoc emits
    // `ASSIGN p = NONDET(ptr_t)` for each modified pointer. Under
    // --add-symex-value-sets symex constrains the post-havoc value
    // to remain inside the pre-havoc points-to set, so the
    // dereferences are sound (they hit the same memory). But the
    // *contents* of that memory are still whatever the pre-loop
    // state placed there — the havoc doesn't randomise the memory
    // the pointers refer to. For loops whose termination depends on
    // what's read through those pointers (e.g. strrchr's
    // `while (*t != 0) ++t` walking a NONDET-length string), IS
    // sees a state where `*t != 0` for every k it unwinds and
    // reports non-termination spuriously.
    //
    // Pre-merge HEAD had a dedicated `has_pointer_only_loop`
    // function gating IS for exactly this shape; the upstream
    // k-induction merge removed it on the claim that the value-set
    // assume made it unnecessary. The assume protects pointer
    // *aliasing* but not pointed-to-data, so the gate is still
    // needed.
    for (const auto &v : modified)
    {
      if (!is_pointer_type(v->type))
        return false;
    }
    return true;
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
  //   (b) The body writes through dereferences (`*p = ...`) or array
  //       indices. collect_lhs_symbols classifies the pointee as
  //       written but the symbol set tracks only the *pointer* (as
  //       read), so modified ends up empty. Without a havoc, IS sees
  //       only the concrete iterates 1..k of the loop and a marker-
  //       unreachable UNSAT within k unwindings is *not* a real non-
  //       termination witness — at higher k the loop may exit (e.g.
  //       termination9's binary counter).
  //
  // We can distinguish (a) from (b) by checking whether the body
  // contains any ASSIGN at all. (a) has none; (b) has at least one.
  // Only (b) is the soundness hazard.
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  for (auto p = loop_head; p != std::next(loop_exit); ++p)
  {
    if (p->is_assign())
      return true;
  }
  return false;
}

/// Place an ASSERT(false) on every edge that leaves a loop,
/// preserving each edge's original successor so CFG semantics are
/// unchanged.
///
/// Loop identification uses a fresh goto_loopst over the post-
/// transformation function body (the natural-loop notion is stable
/// under the havoc + entry-cond ASSUME inserts done by k-induction).
/// For each loop:
///
///   1. Walk the loop body and collect every forward GOTO whose
///      target sits outside [loop_head, loop_exit] — these are the
///      genuine exit edges (top-test, break, early-return-as-goto,
///      goto-label-outside-loop, switch-case branches that escape).
///   2. Group exits by their original target. For each unique target
///      T, insert an ASSERT(false); GOTO T immediately before T, and
///      retarget every exit edge that originally went to T to point
///      at that marker. This preserves each edge's successor
///      exactly.
///   3. For do-while loops (conditional back-edge), the natural
///      fall-through past the back-edge is also an exit. Splice an
///      ASSERT(false) at std::next(loop_exit); the marker falls
///      through to its original successor and so preserves semantics
///      for that path too.
///
/// The retarget (rather than inserting at the original target) is
/// critical: exit-target labels are shared with outer branches that
/// converge there (e.g. `if (c == 0) goto L_after; while (...) goto
/// L_after;`). Inserting at the target would let the outer path also
/// hit the marker, defeating per-loop isolation.
///
/// Marker is flagged inductive_step_instruction so BC/FC skip it;
/// only IS sees the claim.
void insert_markers_for_function(
  const irep_idt &function_name,
  goto_functionst &goto_functions,
  goto_functiont &goto_function)
{
  goto_programt &body = goto_function.body;
  goto_loopst loops(function_name, goto_functions, goto_function);
  if (loops.get_loops().empty())
    return;

  // Snapshot loop boundaries before mutating the body. Sort inner
  // loops first (back-edge location_number descending) so the marker
  // pass processes nearer loops first and outer-loop iterator math
  // stays stable.
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
    infos.begin(), infos.end(), [](const loop_info &a, const loop_info &b) {
      return a.back->location_number > b.back->location_number;
    });

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
    // Also collect RETURN instructions inside the loop range. A return
    // leaves the *loop-owning* function, so it is a genuine loop exit
    // (e.g. `while (1) { x = nondet(); if (x == 0) return 0; }` exits
    // when x == 0). Because insert_markers_for_function processes one
    // function at a time and only scans this loop's own body range, a
    // RETURN found here always belongs to the loop-owning function — a
    // return inside a callee invoked from the loop lives in a separate
    // function body and is never seen here, so we never mistake a
    // callee return for a loop exit.
    std::vector<goto_programt::targett> return_exits;
    for (auto p = loop_head; p != std::next(loop_exit); ++p)
    {
      if (p->is_return())
      {
        return_exits.push_back(p);
        continue;
      }
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
      // jumping directly to orig_target for any caller that reaches
      // the marker.
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

    // Return-exit markers: a RETURN inside the loop body leaves the
    // loop-owning function, so place an ASSERT(false) immediately
    // before each. The marker falls through to the RETURN, preserving
    // semantics, and exposes the return path to IS — without it,
    // `while (1) { ...; if (cond) return; }` shapes have no reachable
    // marker on their real exit and IS reports spurious non-
    // termination.
    for (auto r : return_exits)
    {
      goto_programt::instructiont marker;
      marker.type = ASSERT;
      marker.guard = gen_false_expr();
      marker.location = r->location;
      marker.location.comment("termination per-loop marker");
      marker.function = function_name;
      marker.inductive_step_instruction = true;
      body.instructions.insert(r, marker);
    }

    // Do-while fall-through: a conditional back-edge with a non-
    // trivially-true guard implies control falls through past it on
    // loop exit. Place an ASSERT(false) at std::next(loop_exit); it
    // falls through to its original successor so semantics are
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
/// Set of clang-mangled names of functions that unconditionally
/// terminate the program. Used by the termination walker to decide
/// that a FUNCTION_CALL is an exit instruction (so any guard that
/// reaches it is an exit guard).
using abort_sett = std::unordered_set<irep_idt, irep_id_hash>;

/// Seed the Aborts set with libc / SV-COMP terminators. Each name is
/// inserted twice — once in its bare form and once with the clang
/// mangling — because direct calls in user code carry the mangled
/// form (`c:@F@abort`) but some calls (e.g. from operational models)
/// may use the bare form.
void seed_abort_set(abort_sett &aborts)
{
  static const char *const seeds[] = {
    "abort",
    "exit",
    "_Exit",
    "__assert_fail",
    "__VERIFIER_error",
  };
  for (const char *s : seeds)
  {
    aborts.insert(irep_idt(s));
    aborts.insert(irep_idt(std::string("c:@F@") + s));
  }
}

/// Returns true iff @p function's body unconditionally calls into an
/// Aborts callee along every reachable path (no IF, no backward GOTO,
/// no return). Conservative — used to grow the Aborts set with user-
/// level wrappers (e.g. `reach_error()` when its body is still
/// `__assert_fail(...)`).
bool body_unconditionally_aborts(
  const goto_functiont &function,
  const abort_sett &aborts)
{
  if (!function.body_available)
    return false;
  bool saw_abort = false;
  for (const auto &inst : function.body.instructions)
  {
    if (inst.is_end_function())
      return saw_abort;
    if (
      inst.is_location() || inst.is_skip() || inst.is_decl() ||
      inst.type == DEAD)
      continue;
    if (inst.is_goto())
      return false; // any branching disqualifies
    if (inst.is_return() || inst.is_throw() || inst.is_catch())
      return false;
    if (inst.is_assume() || inst.is_assert())
    {
      if (is_false(inst.guard))
      {
        saw_abort = true;
        continue;
      }
      continue;
    }
    if (inst.is_assign())
      continue;
    if (inst.is_function_call())
    {
      const code_function_call2t &call = to_code_function_call2t(inst.code);
      if (!is_symbol2t(call.function))
        return false;
      if (aborts.count(to_symbol2t(call.function).thename))
      {
        saw_abort = true;
        continue;
      }
      continue; // non-Aborts call: returns normally, body continues
    }
    return false;
  }
  return false;
}

void build_abort_summary(
  const goto_functionst &goto_functions,
  abort_sett &aborts)
{
  seed_abort_set(aborts);
  // Fixed-point: a function whose body unconditionally calls an
  // Aborts function is itself an Aborts function.
  bool changed = true;
  while (changed)
  {
    changed = false;
    for (const auto &kv : goto_functions.function_map)
    {
      if (aborts.count(kv.first))
        continue;
      if (body_unconditionally_aborts(kv.second, aborts))
      {
        aborts.insert(kv.first);
        changed = true;
      }
    }
  }
}

/// Place termination markers immediately before two kinds of
/// FUNCTION_CALL instructions inside a loop body:
///
///   * Direct calls to an Aborts callee (abort/exit/__assert_fail/
///     __VERIFIER_error/user-level wrappers around them): emit
///     ASSERT(false). The call unconditionally exits, so the marker
///     fires unconditionally on the abort path.
///
///   * Calls to `__VERIFIER_assert(cond)`: emit ASSERT(cond). The
///     SV-COMP wrapper aborts iff cond is false, so the marker fires
///     exactly when the abort path would be taken. The call-site
///     argument is already expressed in caller-local terms, so no
///     parameter substitution is needed — we don't inspect the
///     wrapper's body at all. The wrapper's SV-COMP definition is
///     fixed; recognising it by name is sound across the entire
///     benchmark set.
///
/// Without these markers, programs whose only real loop exit is via
/// the call would produce a vacuous IS counterexample — the natural-
/// exit marker placed by insert_markers_for_function lives past the
/// loop's `IF !1 GOTO exit`, statically unreachable from the back
/// edge. IS would see no marker on any unwinding and report non-
/// termination.
///
/// The original FUNCTION_CALL is left in place; the marker just
/// precedes it. Markers are flagged inductive_step_instruction so
/// BC/FC skip them, matching the natural-exit markers.
void insert_abort_call_markers_for_function(
  const irep_idt &function_name,
  goto_functionst &goto_functions,
  goto_functiont &goto_function,
  const abort_sett &aborts)
{
  // Conditional-abort wrappers: functions of shape
  //   void name(T cond) { if (!cond) abort(); }
  // SV-COMP and its derivative benchmarks ship several such wrappers
  // under different names. We recognise them by name and emit
  // ASSERT(cond) before each call. Both bare and clang-mangled forms
  // are listed to match the convention in seed_abort_set.
  const abort_sett conditional_abort_wrappers = {
    "__VERIFIER_assert",
    "c:@F@__VERIFIER_assert",
    "assume_abort_if_not",
    "c:@F@assume_abort_if_not",
    "assert",
    "c:@F@assert",
  };

  goto_programt &body = goto_function.body;
  goto_loopst loops(function_name, goto_functions, goto_function);
  if (loops.get_loops().empty())
    return;

  // Snapshot per-loop ranges so mid-list inserts don't invalidate the
  // iteration. Sort inner loops first (back-edge location_number
  // descending) so outer iterator math stays stable.
  struct loop_range
  {
    goto_programt::targett head;
    goto_programt::targett back;
  };
  std::vector<loop_range> ranges;
  for (const auto &loop : loops.get_loops())
    ranges.push_back(
      {loop.get_original_loop_head(), loop.get_original_loop_exit()});
  std::sort(
    ranges.begin(), ranges.end(), [](const loop_range &a, const loop_range &b) {
      return a.back->location_number > b.back->location_number;
    });

  // Each mark records the call to precede and the guard expression to
  // assert. A nil guard means "ASSERT(false)" (unconditional, for
  // direct Aborts calls); any other guard becomes "ASSERT(guard)".
  struct mark_pointt
  {
    goto_programt::targett at;
    expr2tc guard;
  };

  for (const auto &r : ranges)
  {
    std::vector<mark_pointt> marks;
    for (auto p = r.head; p != std::next(r.back); ++p)
    {
      if (!p->is_function_call())
        continue;
      const code_function_call2t &call = to_code_function_call2t(p->code);
      if (!is_symbol2t(call.function))
        continue;
      const irep_idt &callee = to_symbol2t(call.function).thename;

      if (aborts.count(callee))
      {
        marks.push_back({p, expr2tc()}); // unconditional ASSERT(false)
        continue;
      }

      // Conditional-abort wrappers: shape `void name(T cond) { if
      // (!cond) abort(); }`. SV-COMP and its derivative benchmarks
      // ship several such wrappers under different names — recognise
      // them by name and emit ASSERT(cond) before each call. The
      // call-site argument is already the caller's expression for
      // cond, so no parameter substitution is needed. For integer
      // cond (the typical signature in SV-COMP), coerce to bool via
      // `cond != 0` so the assertion guard has bool type downstream
      // in the solver.
      if (conditional_abort_wrappers.count(callee))
      {
        if (call.operands.empty() || is_nil_expr(call.operands[0]))
          continue;
        expr2tc guard = call.operands[0];
        if (!is_bool_type(guard->type))
          guard = notequal2tc(guard, gen_zero(guard->type));
        marks.push_back({p, guard});
      }
    }

    for (const auto &m : marks)
    {
      goto_programt::instructiont marker;
      marker.type = ASSERT;
      if (is_nil_expr(m.guard))
        marker.guard = gen_false_expr();
      else
        marker.guard = m.guard;
      marker.location = m.at->location;
      marker.location.comment("termination abort-call marker");
      marker.function = function_name;
      marker.inductive_step_instruction = true;
      body.instructions.insert(m.at, marker);
    }
  }
}

} // namespace

void goto_termination(goto_functionst &goto_functions, optionst &options)
{
  // Build the Aborts summary first so insert_abort_call_markers_for_
  // function can recognise abort/_Exit/__assert_fail and user-level
  // wrappers around them.
  abort_sett aborts;
  build_abort_summary(goto_functions, aborts);

  // Apply k-induction's per-loop havoc + assume-entry-cond to every
  // function uniformly. __ESBMC_main is loop-free so this is a no-op
  // there; library helpers (body.hide) DO have loops but those are
  // explicitly skipped in the per-function pass below.
  goto_k_induction(goto_functions);

  // Per-function pass: inject no-op-cycle ASSUMEs, gate IS for loops
  // that would make IS UNSAT unsound, and insert termination
  // markers. Library helpers (body.hide) are skipped — their per-
  // loop markers would be reachable from every program via
  // stdlib/pthread chains and defeat IS-UNSAT discrimination, and
  // their havoc'd loops shouldn't trigger the program-wide IS gate.
  bool any_unreliable_is_loop = false;
  Forall_goto_functions (it, goto_functions)
  {
    if (!it->second.body_available)
      continue;
    if (it->first == "__ESBMC_main")
      continue;

    // For each loop: inject no-op-cycle witnesses + check the IS
    // gate. Both use goto_loopst's natural-loop notion, which is
    // stable across kinduction's rewrites.
    //
    // The IS gate (loop_is_is_unreliable) and no-op-cycle witnesses
    // apply to USER loops only. Library helpers (body.hide) are
    // verification-harness scaffolding — e.g. __ESBMC_atexit_handler's
    // `while (atexit_count > 0)` loop, linked into every program,
    // classifies as the empty-modified-set-with-assign hazard (case b
    // in loop_is_is_unreliable) and would flip the program-wide IS
    // gate to "unreliable" for *every* program, disabling IS globally
    // and making non-termination unprovable. Their termination is
    // never the reason a user program loops forever, so they must not
    // contribute to the gate. (The marker pass below DOES run on them,
    // so a program whose only loop is a library call still gets a
    // termination claim.)
    if (!it->second.body.hide)
    {
      goto_loopst loops(it->first, goto_functions, it->second);
      for (auto &loop : loops.get_loops())
      {
        if (!loop.get_modified_loop_vars().empty())
          inject_noop_cycle_assumes(it->second, loop);
        if (loop_is_is_unreliable(loop))
          any_unreliable_is_loop = true;
      }
    }

    // Marker pass: needs its own fresh goto_loopst (the loop list is
    // sorted innermost-first internally for stable iterator math).
    // Runs on library helpers too, so a program whose only loop lives
    // inside a hidden operational model (e.g. main() { memset(...); })
    // still has a termination claim for IS/FC to discharge.
    insert_markers_for_function(it->first, goto_functions, it->second);

    // Abort-call markers: precede every direct call to an Aborts
    // function (abort/exit/__assert_fail/...) inside a loop body
    // with an ASSERT(false). This exposes abort paths to IS so the
    // marker is reachable along the abort flow, instead of being
    // stranded at the natural loop exit (which is statically
    // unreachable for `while(1){... abort();}` shapes).
    insert_abort_call_markers_for_function(
      it->first, goto_functions, it->second, aborts);
  }

  goto_functions.update();

  if (any_unreliable_is_loop)
    options.set_option("disable-inductive-step", true);
}