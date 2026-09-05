
/*
 * Loop invariant instrumentation pass.
 *
 * Converts __ESBMC_loop_invariant(I) annotations into a three-part inductive
 * invariant check following the standard k-induction schema:
 *
 *   1. Base case    -- ASSERT I before the loop; verifies I holds on entry.
 *
 *   2. Havoc + Assume -- Loop-modified variables are set to nondet values,
 *                        then I is ASSUMEd, modelling an arbitrary iteration.
 *
 *   3. Inductive step -- After one loop body iteration, ASSERT I to verify
 *                        the body preserves the invariant.  ASSUME(false) then
 *                        terminates this symbolic execution path.
 *
 * Function calls inside the invariant (e.g., in_range(x)) require the
 * DECL/FUNCTION_CALL pair to be re-evaluated after the havoc step so that the
 * return value reflects the new nondet values.  extract_and_remove_side_effects
 * moves these instructions into a side_effects block that is re-inserted before
 * each ASSERT/ASSUME.  The Clang frontend avoids losing guard conditions in
 * A && f(x) && B by special-casing __ESBMC_loop_invariant in goto_sideeffects
 * (flatten and, remove_sideeffects per conjunct, rebuild and), so the stored
 * invariant is always the full expression and no short-circuit recovery is
 * needed here.
 *
 * When --loop-frame-rule is enabled, the havoc step is enhanced with:
 *   Snapshot -> Havoc -> FrameRule(Assume unchanged == snapshot) -> Assume invariants
 * This preserves the relationship between modified and unmodified variables.
 */

#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/frame_enforcer.h>
#include <goto-programs/remove_no_op.h>
#include <util/lang/c_types.h>
#include <util/expr/expr_util.h>
#include <util/base/i2string.h>
#include <util/irep/std_expr.h>
#include <util/config/options.h>
#include <irep2/irep2_utils.h>
#include <functional>
#include <map>

// ---------------------------------------------------------------------------
// Shared helper: extract the loop invariant near a loop head
// ---------------------------------------------------------------------------

/// Maximum number of instructions to search backwards from the loop head
/// when locating the LOOP_INVARIANT instruction.
static constexpr size_t kMaxInvariantSearchBack = 10;

/// Walk backwards from @p loop_head (up to kMaxInvariantSearchBack steps) and
/// return the invariant expression(s) for the nearest LOOP_INVARIANT found.
/// Multiple conjuncts are folded into a single and2tc.  Returns an empty
/// vector when no annotated LOOP_INVARIANT is in range.
static std::vector<expr2tc> extract_invariants_near(
  goto_programt::targett loop_head,
  goto_programt::targett begin)
{
  std::vector<expr2tc> invariants;

  if (loop_head == begin)
    return invariants;

  goto_programt::targett it = loop_head;
  size_t dist = 0;

  // Fold a LOOP_INVARIANT instruction's expression list into `invariants`.
  // Multiple sub-expressions are combined with &&.
  auto collect = [&](const std::list<expr2tc> &lst) {
    if (lst.size() == 1)
    {
      invariants.push_back(lst.front());
    }
    else
    {
      auto jt = lst.begin();
      expr2tc combined = *jt;
      for (++jt; jt != lst.end(); ++jt)
        combined = and2tc(combined, *jt);
      invariants.push_back(combined);
    }
  };

  // Returns true if the instruction is a compiler-generated DECL, ASSIGN, or
  // FUNCTION_CALL (name contains '$', e.g. return_value$___ESBMC_forall$N).
  // These may legitimately appear between consecutive __ESBMC_loop_invariant()
  // calls.  FUNCTION_CALL instructions arise when remove_function_call emits a
  // DECL + FUNCTION_CALL pair for helper functions called inside an invariant.
  auto is_compiler_temp = [](goto_programt::const_targett t) -> bool {
    if (t->is_decl() && is_code_decl2t(t->code))
      return id2string(to_code_decl2t(t->code).value).find('$') !=
             std::string::npos;
    if (t->is_assign() && is_code_assign2t(t->code))
    {
      const auto &assign = to_code_assign2t(t->code);
      return is_symbol2t(assign.target) &&
             id2string(to_symbol2t(assign.target).thename).find('$') !=
               std::string::npos;
    }
    if (t->is_function_call() && is_code_function_call2t(t->code))
    {
      const auto &call = to_code_function_call2t(t->code);
      return !is_nil_expr(call.ret) && is_symbol2t(call.ret) &&
             id2string(to_symbol2t(call.ret).thename).find('$') !=
               std::string::npos;
    }
    return false;
  };

  // Phase 1: skip non-invariant instructions (including for-init assignments)
  // until we find the closest LOOP_INVARIANT for this loop.
  while (it != begin && dist < kMaxInvariantSearchBack)
  {
    --it;
    ++dist;

    if (!it->is_loop_invariant())
      continue;

    const std::list<expr2tc> &inv_list = it->get_loop_invariants();
    if (inv_list.empty())
      continue;

    collect(inv_list);

    // Phase 2: collect any additional LOOP_INVARIANT instructions that belong
    // to the same loop.  These arise when the user writes multiple separate
    // __ESBMC_loop_invariant() calls (issue #3936).  The only instructions
    // that may appear between consecutive same-loop invariants are compiler-
    // generated temporaries; any real instruction means we have crossed into
    // a different context (e.g. outer-loop body), so we stop immediately.
    // Reuse `dist` so that Phase 1 + Phase 2 together respect the same
    // kMaxInvariantSearchBack bound and avoid an O(n) scan in large functions.
    while (it != begin && dist < kMaxInvariantSearchBack)
    {
      --it;
      ++dist;
      if (it->is_loop_invariant())
      {
        const std::list<expr2tc> &extra = it->get_loop_invariants();
        if (!extra.empty())
          collect(extra);
        continue;
      }
      if (is_compiler_temp(it))
        continue;
      break; // real instruction — stop clustering
    }
    break; // done: found the invariant group for this loop
  }

  return invariants;
}

void goto_loop_invariant(
  goto_functionst &goto_functions,
  contextt &context,
  bool use_frame_rule)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_loop_invariantt(
        it->first, goto_functions, it->second, context, use_frame_rule);

  goto_functions.update();
}

static void
collect_symbols_local(const expr2tc &expr, std::set<irep_idt> &symbols);

/// True when some symbol the loop's guard reads is one the havoc covers. When
/// nothing the guard reads is havoc'd, the exit edge is decided by the
/// concrete pre-loop state and cannot change, so the claims after the loop are
/// never reached rather than checked (issue #7478).
static bool guard_reads_havocd_var(
  goto_programt::targett head,
  const goto_functiont &fn,
  const loopst &loop)
{
  std::set<irep_idt> havocd_names;
  for (const auto &v : loop.get_modified_loop_vars())
    if (is_symbol2t(v))
      havocd_names.insert(to_symbol2t(v).get_symbol_name());

  for (goto_programt::targett it = head; it != fn.body.instructions.end(); ++it)
  {
    std::set<irep_idt> read;
    collect_symbols_local(it->code, read);
    collect_symbols_local(it->guard, read);
    for (const auto &name : read)
      if (havocd_names.count(name))
        return true;
    if (it->is_goto())
      break;
  }
  return false;
}

void goto_loop_invariantt::goto_loop_invariant()
{
  for (auto &function_loop : function_loops)
  {
    // If the loop modifies no variables the havoc step is empty and the
    // inductive check is trivially true.  We still process the loop so that
    // the base-case ASSERT is emitted (it can still catch a wrong invariant).
    // TODO: consider whether an empty modified-vars set warrants a warning.
    convert_loop_with_invariant(function_loop);
  }
}

void goto_loop_invariantt::convert_loop_with_invariant(loopst &loop)
{
  // Get current loop head and loop exit
  goto_programt::targett loop_head = loop.get_original_loop_head();

  // Extract loop invariants from LOOP_INVARIANT instructions
  std::vector<expr2tc> invariants = extract_loop_invariants(loop);

  if (invariants.empty())
    return;

  log_status(
    "Processing {} loop invariant{}",
    invariants.size(),
    invariants.size() == 1 ? "" : "s");

  // Extract loop assigns targets (for frame rule)
  std::vector<expr2tc> loop_assigns = extract_loop_assigns(loop);

  // Extract invariant-related DECL/FUNCTION_CALLs and re-insert before each
  // ASSERT/ASSUME.
  goto_programt side_effects;
  extract_and_remove_side_effects(loop_head, loop, invariants, side_effects);

  // insert_assert_before_loop swaps the base case into the loop-head slot and
  // moves the head's own instruction to a fresh node behind it, so the
  // `loop_head` iterator no longer denotes the loop head afterwards. Anchor the
  // instruction that follows it, which no insertion before the head can move.
  goto_programt::targett after_head = std::next(loop_head);

  // 1. Insert ASSERT invariant before loop (base case)
  insert_assert_before_loop(loop_head, invariants, side_effects);

  // A write through a dereference has no named symbol for the havoc to cover.
  // When the guard reads nothing the havoc does reach, the loop is left at its
  // concrete pre-loop state: the exit edge cannot change, the invariant is
  // discharged against one concrete iteration and every claim after the loop is
  // dropped without being solved (issue #7478). Leave the loop for the unwinder
  // rather than report a proof we did not make. The base case above is checked
  // against the concrete pre-loop state and needs no havoc, so it still stands.
  if (
    loop.writes_through_pointer() &&
    !guard_reads_havocd_var(std::prev(after_head), goto_function, loop))
  {
    log_warning(
      "loop invariant at {} not checked beyond its base case: the loop writes "
      "through a pointer the havoc cannot cover",
      loop.get_original_loop_head()->location.as_string());
    return;
  }

  // 2. Insert HAVOC and ASSUME at the loop head, ahead of the guard
  insert_havoc_and_assume_before_condition(
    std::prev(after_head), loop, invariants, loop_assigns, side_effects);

  // 3. Insert inductive step verification and loop termination
  insert_inductive_step_and_termination(loop, invariants, side_effects);

  // Cleanup: free the frame enforcer after both steps are done.
  delete active_frame_enforcer;
  active_frame_enforcer = nullptr;
  active_loop_assigns.clear();
}

std::vector<expr2tc>
goto_loop_invariantt::extract_loop_invariants(const loopst &loop)
{
  return extract_invariants_near(
    loop.get_original_loop_head(), goto_function.body.instructions.begin());
}

std::vector<expr2tc>
goto_loop_invariantt::extract_loop_assigns(const loopst &loop)
{
  std::vector<expr2tc> assigns;

  goto_programt::targett loop_head = loop.get_original_loop_head();

  if (loop_head == goto_function.body.instructions.begin())
    return assigns;

  // Search backwards from loop head to find LOOP_INVARIANT with assigns targets
  goto_programt::targett search_it = loop_head;
  size_t search_distance = 0;

  while (search_it != goto_function.body.instructions.begin() &&
         search_distance < kMaxInvariantSearchBack)
  {
    --search_it;
    ++search_distance;

    if (search_it->is_loop_invariant())
    {
      auto const &targets = search_it->get_loop_assigns_targets();
      for (const auto &target : targets)
      {
        assigns.push_back(target);
      }
      break;
    }
  }

  return assigns;
}

void goto_loop_invariantt::collect_symbols(
  const expr2tc &expr,
  std::set<irep_idt> &symbols)
{
  if (!expr)
    return;

  if (is_symbol2t(expr))
  {
    symbols.insert(to_symbol2t(expr).get_symbol_name());
    return;
  }

  for (size_t i = 0; i < expr->get_num_sub_exprs(); ++i)
  {
    const expr2tc *sub = expr->get_sub_expr(i);
    if (sub && *sub)
      collect_symbols(*sub, symbols);
  }
}

/// Return true if \p expr is a trivial RHS (constant 0/1, constant bool, or
/// NONDET). We skip collecting such assignments when tracing back from the
/// invariant so that we only pull in the "real" definition (e.g. from a
/// function call) and not the fallback or initialiser.
static bool is_trivial_rhs(const expr2tc &expr)
{
  if (!expr)
    return true;
  if (is_constant_int2t(expr))
  {
    const auto &v = to_constant_int2t(expr).value;
    return v == 0 || v == 1;
  }
  if (is_constant_bool2t(expr))
    return true;
  if (
    is_sideeffect2t(expr) &&
    to_sideeffect2t(expr).kind == sideeffect2t::allockind::nondet)
    return true;
  return false;
}

/// Heuristic: compiler-generated temporaries (e.g. for short-circuit evaluation)
/// typically have '$' in their name; user variables do not. Used to avoid
/// moving user variable DECLs/ASSIGNs while still collecting compiler temps.
static bool is_likely_compiler_temp(const irep_idt &id)
{
  return id2string(id).find('$') != std::string::npos;
}

/// True if \p full_or_short matches \p decl_id: either exact match, or
/// (when \p allow_suffix) \p full_or_short ends with "::" + decl_id.
static bool symbol_name_matches(
  const irep_idt &full_or_short,
  const irep_idt &decl_id,
  bool allow_suffix)
{
  const std::string s = id2string(full_or_short);
  const std::string d = id2string(decl_id);
  if (s == d)
    return true;
  if (!allow_suffix)
    return false;
  if (s.size() < d.size() + 2)
    return false;
  const std::size_t suffix_pos = s.size() - d.size();
  return s[suffix_pos - 2] == ':' && s[suffix_pos - 1] == ':' &&
         s.compare(suffix_pos, d.size(), d) == 0;
}

/// Local helper: collect all symbol names reachable from an expression tree.
static void
collect_symbols_local(const expr2tc &expr, std::set<irep_idt> &symbols)
{
  if (!expr)
    return;

  if (is_symbol2t(expr))
  {
    symbols.insert(to_symbol2t(expr).get_symbol_name());
    return;
  }

  for (size_t i = 0; i < expr->get_num_sub_exprs(); ++i)
  {
    const expr2tc *sub = expr->get_sub_expr(i);
    if (sub && *sub)
      collect_symbols_local(*sub, symbols);
  }
}

/// Shared implementation used by both the legacy loop-invariant pass and the
/// combined loop-invariant + k-induction pass.  See the declaration of
/// goto_loop_invariantt::extract_and_remove_side_effects for a high-level
/// description; this helper just takes an explicit goto_function parameter so
/// it can be reused from both passes.
static void extract_and_remove_side_effects_impl(
  goto_functiont &goto_function,
  goto_programt::targett loop_head,
  const loopst &loop,
  const std::vector<expr2tc> &invariants,
  goto_programt &side_effects_out)
{
  // Collect symbol names referenced in the invariants.
  std::set<irep_idt> inv_symbols;
  for (const auto &inv : invariants)
    collect_symbols_local(inv, inv_symbols);

  if (inv_symbols.empty())
    return;

  // Names of variables modified in the loop: do not move their DECLs (they
  // are user variables like i, sum, y; moving them would break base case).
  std::set<irep_idt> modified_in_loop;
  for (const auto &e : loop.get_modified_loop_vars())
    if (is_symbol2t(e))
      modified_in_loop.insert(to_symbol2t(e).get_symbol_name());

  auto decl_modified_in_loop = [&modified_in_loop](const irep_idt &decl_val) {
    for (const auto &sym : modified_in_loop)
      if (symbol_name_matches(sym, decl_val, true))
        return true;
    return false;
  };

  // Find the LOOP_INVARIANT instruction preceding loop_head.
  goto_programt::targett loop_inv_it = loop_head;
  size_t back_dist = 0;
  while (loop_inv_it != goto_function.body.instructions.begin() &&
         back_dist < kMaxInvariantSearchBack)
  {
    --loop_inv_it;
    ++back_dist;
    if (loop_inv_it->is_loop_invariant())
      break;
  }
  if (!loop_inv_it->is_loop_invariant())
    return;

  // Walk backwards from LOOP_INVARIANT: collect instructions that define
  // temporaries used in the invariant. dep_symbols = symbols the invariant
  // (or already-collected definitions) depend on. Also trace through ASSIGNs
  // so when the invariant uses a compiler temporary (e.g. tmp$2) assigned from
  // a function-call result, we collect that ASSIGN and the FUNCTION_CALL/DECL
  // (fixes issue #3711: function call in the middle of invariant body).
  // Collect DECL only when the symbol is in dep_symbols and not modified in
  // the loop (so we do not move DECLs for user variables like i, sum).
  std::vector<goto_programt::targett> to_remove;
  std::set<irep_idt> dep_symbols = inv_symbols;
  std::set<irep_idt> fc_return_syms;

  goto_programt::targett search = loop_inv_it;
  while (search != goto_function.body.instructions.begin())
  {
    --search;

    if (search->is_assign() && is_code_assign2t(search->code))
    {
      const code_assign2t &assign = to_code_assign2t(search->code);
      if (!is_symbol2t(assign.target))
        break;
      irep_idt lhs_sym = to_symbol2t(assign.target).get_symbol_name();
      if (dep_symbols.count(lhs_sym) == 0)
        break;
      if (!is_likely_compiler_temp(lhs_sym))
        break;
      if (is_trivial_rhs(assign.source))
        continue;
      to_remove.push_back(search);
      collect_symbols_local(assign.source, dep_symbols);
      continue;
    }

    if (search->is_function_call())
    {
      if (!is_code_function_call2t(search->code))
        break;
      const code_function_call2t &fc = to_code_function_call2t(search->code);
      if (!is_nil_expr(fc.ret) && is_symbol2t(fc.ret))
      {
        irep_idt sym = to_symbol2t(fc.ret).get_symbol_name();
        if (dep_symbols.count(sym))
        {
          fc_return_syms.insert(sym);
          to_remove.push_back(search);
          continue;
        }
      }
      break;
    }

    if (search->is_decl())
    {
      if (!is_code_decl2t(search->code))
        break;
      const code_decl2t &decl = to_code_decl2t(search->code);
      if (decl_modified_in_loop(decl.value))
      {
        continue;
      }
      const bool allow_suffix = is_likely_compiler_temp(decl.value);
      bool is_fc_ret = false;
      bool is_extra_dep = false;
      for (const auto &sym : fc_return_syms)
        if (symbol_name_matches(sym, decl.value, allow_suffix))
        {
          is_fc_ret = true;
          break;
        }
      if (!is_fc_ret)
        for (const auto &sym : dep_symbols)
          if (
            !inv_symbols.count(sym) &&
            symbol_name_matches(sym, decl.value, allow_suffix))
          {
            is_extra_dep = true;
            break;
          }
      if (is_fc_ret || is_extra_dep)
      {
        to_remove.push_back(search);
        continue;
      }
      if (inv_symbols.count(decl.value) && !allow_suffix)
        break;
      continue;
    }

    if (search->is_goto())
    {
      continue;
    }

    // Consecutive __ESBMC_loop_invariant() calls for the same loop lower to
    // several adjacent LOOP_INVARIANT instructions (issue #3936), separated
    // only by the DECL/ASSIGN that compute each invariant's temporaries. Skip
    // past an earlier LOOP_INVARIANT so we keep collecting the side effects of
    // *every* invariant in the cluster, not just the nearest one; otherwise a
    // quantifier invariant that is not the last in the cluster is never
    // re-evaluated after havoc and its stale pre-loop value is assumed.
    if (search->is_loop_invariant())
      continue;

    break;
  }

  if (to_remove.empty())
    return;

  for (auto rit = to_remove.rbegin(); rit != to_remove.rend(); ++rit)
  {
    side_effects_out.instructions.push_back(**rit);
    /* Erasing dangles any goto that targets this instruction -- a preceding
     * loop's exit can land on one of the invariant's temporaries, and
     * compute_target_numbers then aborts on the unnumbered target. Leaving a
     * skip keeps every incoming target valid; only DECL/ASSIGN/FUNCTION_CALL
     * reach here, so nothing that carries targets of its own is cleared. */
    (*rit)->make_skip();
  }
}

void goto_loop_invariantt::extract_and_remove_side_effects(
  goto_programt::targett loop_head,
  const loopst &loop,
  const std::vector<expr2tc> &invariants,
  goto_programt &side_effects_out)
{
  extract_and_remove_side_effects_impl(
    goto_function, loop_head, loop, invariants, side_effects_out);
}

void goto_loop_invariantt::insert_assert_before_loop(
  goto_programt::targett &loop_head,
  const std::vector<expr2tc> &invariants,
  const goto_programt &side_effects)
{
  goto_programt dest;

  for (const auto &instr : side_effects.instructions)
    dest.instructions.push_back(instr);

  for (const auto &invariant : invariants)
  {
    // Create assert instruction for each invariant
    goto_programt::targett t = dest.add_instruction(ASSERT);
    t->guard = invariant;
    t->location = loop_head->location;
    t->location.comment("loop invariant base case");
    t->location.property("invariant-base-case");
  }

  // Insert before the loop head
  goto_function.body.insert_swap(loop_head, dest);
}

void goto_loop_invariantt::insert_havoc_and_assume_before_condition(
  goto_programt::targett loop_head,
  const loopst &loop,
  const std::vector<expr2tc> &invariants,
  const std::vector<expr2tc> &loop_assigns,
  goto_programt &side_effects)
{
  // The havoc models an arbitrary iteration, so it belongs at the loop head,
  // where the base case and the inductive step already anchor the invariant --
  // not at the guard's IF. A guard with a side effect (`while (cnt--)`), a call
  // (`while (f(&x))`) or a short circuit is lowered to instructions that sit
  // between the two; havoc'ing after them leaves the IF testing a pre-havoc
  // temporary, which makes the loop-exit edge infeasible and silently drops
  // every claim after the loop (issue #7478).
  goto_programt::targett insert_point = loop_head;

  goto_programt dest;

  auto const &loop_vars = loop.get_modified_loop_vars();

  // =========================================================
  // Frame Rule Step 1: Materialize Snapshots (if enabled)
  // Capture pre-state before havoc for frame rule enforcement.
  // The snapshots are also used in the inductive step for ASSERT compliance.
  // =========================================================
  if (use_frame_rule)
  {
    active_frame_enforcer = new frame_enforcert(context);
    active_loop_assigns = loop_assigns;

    std::vector<expr2tc> vars_to_snapshot;
    for (const auto &v : loop_vars)
      vars_to_snapshot.push_back(v);

    active_frame_enforcer->materialize_snapshots(
      vars_to_snapshot, dest, loop_head->location, "loop");

    // Patch old_snapshot side-effect assignments to use the frame snapshots.
    // Replaces: return_value$___ESBMC_old_raw$N = old_snapshot(var)
    // With:     return_value$___ESBMC_old_raw$N = (void*)&snapshot_of_var
    // This also propagates to insert_inductive_step_and_termination which is
    // called after this function and receives the same side_effects object.
    active_frame_enforcer->patch_old_snapshot_assigns(side_effects);
  }

  // =========================================================
  // Step 2: Standard Havoc — assign nondet to all modified variables
  // =========================================================
  for (auto const &lhs : loop_vars)
  {
    // do not assign nondeterministic value to pointers if we assume
    // objects extracted from the value set analysis
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(lhs))
      continue;

    // Generate a nondeterministic value for the loop variable
    expr2tc rhs = gen_nondet(lhs->type);

    // Create an assignment instruction for the nondeterministic value
    goto_programt::targett t = dest.add_instruction(ASSIGN);
    t->code = code_assign2tc(lhs, rhs);
    t->location = loop_head->location;
    t->location.comment("loop invariant havoc");
    t->loop_invariant_havoc = true;
  }

  // =========================================================
  // Frame Rule Step 3: Enforce Frame Conditions (if enabled)
  // ASSUME: for vars NOT in assigns set, assume var == snapshot (k-induction hypothesis)
  // =========================================================
  if (use_frame_rule && active_frame_enforcer && !loop_assigns.empty())
  {
    active_frame_enforcer->enforce_frame_rule(
      loop_assigns, dest, loop_head->location);
  }

  // Symbols defined by the extracted side-effect block (function-call results
  // and their temporaries). A conjunct that reads one of these is assumed
  // *after* the side effects; one that does not is a pure constraint on the
  // havoc'd variables and is assumed *before* them, so it bounds any havoc'd
  // value feeding a side-effecting invariant — e.g. a function call whose body
  // loops over that value. Without this ordering the call is symex'd with an
  // unconstrained nondet bound and its inner loop's unwinding assertion fails
  // spuriously (issue #6189).
  std::set<irep_idt> side_effect_defs;
  for (const auto &instr : side_effects.instructions)
  {
    if (instr.is_assign() && is_code_assign2t(instr.code))
    {
      const code_assign2t &a = to_code_assign2t(instr.code);
      if (is_symbol2t(a.target))
        side_effect_defs.insert(to_symbol2t(a.target).get_symbol_name());
    }
    else if (instr.is_function_call() && is_code_function_call2t(instr.code))
    {
      const code_function_call2t &fc = to_code_function_call2t(instr.code);
      if (!is_nil_expr(fc.ret) && is_symbol2t(fc.ret))
        side_effect_defs.insert(to_symbol2t(fc.ret).get_symbol_name());
    }
  }

  auto uses_side_effect = [&side_effect_defs](const expr2tc &conjunct) {
    std::set<irep_idt> syms;
    collect_symbols_local(conjunct, syms);
    for (const auto &s : syms)
      if (side_effect_defs.count(s))
        return true;
    return false;
  };

  // Flatten top-level conjunctions so a pure conjunct can be split out even when
  // the frontend folded it together with a side-effecting one into `a && b`.
  // The conjunct guards are themselves pure: any embedded call has already been
  // hoisted into the side_effects block (see file header).
  std::function<void(const expr2tc &)> partition;
  std::vector<expr2tc> pure, impure;
  partition = [&](const expr2tc &e) {
    if (is_and2t(e))
    {
      partition(to_and2t(e).side_1);
      partition(to_and2t(e).side_2);
    }
    else
      (uses_side_effect(e) ? impure : pure).push_back(e);
  };
  for (const auto &invariant : invariants)
    partition(invariant);

  auto emit_assume = [&](const expr2tc &conjunct) {
    expr2tc inst = conjunct;
    if (use_frame_rule && active_frame_enforcer)
      inst = active_frame_enforcer->replace_old_with_snapshots(conjunct);

    goto_programt::targett t = dest.add_instruction(ASSUME);
    t->guard = inst;
    t->location = loop_head->location;
    t->location.comment("loop invariant step case");
  };

  // Assume pure conjuncts, run the side effects under those constraints, then
  // assume the side-effect-dependent conjuncts.
  for (const auto &conjunct : pure)
    emit_assume(conjunct);
  for (const auto &instr : side_effects.instructions)
    dest.instructions.push_back(instr);
  for (const auto &conjunct : impure)
    emit_assume(conjunct);

  // Note: active_frame_enforcer is NOT deleted here.
  // It is reused in insert_inductive_step_and_termination for the ASSERT
  // compliance check, then freed in convert_loop_with_invariant.

  // Insert before the loop condition
  goto_function.body.insert_swap(insert_point, dest);
}

void goto_loop_invariantt::insert_inductive_step_and_termination(
  const loopst &loop,
  const std::vector<expr2tc> &invariants,
  const goto_programt &side_effects)
{
  // Insert at the end of loop body (before GOTO back to loop head)
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  goto_programt::targett insert_point = loop_exit;

  goto_programt dest;

  // 1. Emit side-effect instructions (post-iteration values).
  for (const auto &instr : side_effects.instructions)
    dest.instructions.push_back(instr);

  // 2. ASSERT loop assigns compliance (frame rule ASSERT mode).
  // After one loop body iteration, assert that variables NOT in the assigns
  // clause still equal their pre-iteration snapshots.
  // This verifies the user's __ESBMC_loop_assigns annotation is accurate,
  // analogous to the function-contract assigns compliance check (Phase 2A).
  if (use_frame_rule && active_frame_enforcer && !active_loop_assigns.empty())
  {
    active_frame_enforcer->enforce_frame_rule(
      active_loop_assigns, dest, loop_exit->location, frame_modet::ASSERT);
  }

  // A `do`-`while` back edge carries the loop's own exit test, so unlike the
  // unconditional back edge of a `while` or `for` loop it cannot be cut with
  // ASSUME(false): the fall-through is the exit path, and killing it drops
  // every claim after the loop (issue #7478). Assume the negated condition
  // instead and drop the back edge outright, and guard the inductive step by
  // that condition so it is required only of an iteration that would actually
  // follow -- the same correction the combined pass made for do-while in
  // PR #3777.
  const bool conditional_back_edge =
    loop_exit->is_goto() && !is_true(loop_exit->guard);
  const expr2tc exit_cond =
    conditional_back_edge ? not2tc(loop_exit->guard) : gen_false_expr();

  // 3. ASSERT for inductive step.
  for (const auto &invariant : invariants)
  {
    // Create assert instruction for each invariant
    goto_programt::targett t = dest.add_instruction(ASSERT);
    t->guard =
      conditional_back_edge ? expr2tc(or2tc(exit_cond, invariant)) : invariant;
    t->location = loop_exit->location;
    t->location.comment("loop invariant inductive step");
    t->location.property("invariant-inductive-step");
  }

  // 4. Cut the back edge: the havoc already covers every iteration.
  goto_programt::targett t = dest.add_instruction(ASSUME);
  t->guard = exit_cond;
  t->location = loop_exit->location;
  t->location.comment("loop termination");

  // ASSUME(false) leaves an unconditional back edge statically unreachable, but
  // a conditional one still has a satisfiable guard and symex would unwind it
  // forever. Remove it; the ASSUME above already constrains the exit path.
  if (conditional_back_edge)
    loop_exit->make_skip();

  // Insert at the insert point
  goto_function.body.insert_swap(insert_point, dest);
}

// ---------------------------------------------------------------------------
// Combined mode implementation
// ---------------------------------------------------------------------------

void goto_loop_invariant_combined(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_loop_invariant_combinedt(it->first, goto_functions, it->second);

  goto_functions.update();
}

void goto_loop_invariant_combinedt::process_loops_combined()
{
  for (auto &loop : function_loops)
    insert_invariant_verification_branch(loop);
}

void goto_loop_invariant_combinedt::copy_loop_body(
  goto_programt::targett body_begin,
  goto_programt::targett loop_exit,
  goto_programt &out) const
{
  // Build target map for intra-loop jumps (similar to loop_unroll.cpp)
  std::map<goto_programt::targett, unsigned> target_map;
  {
    unsigned count = 0;
    for (auto t = body_begin; t != loop_exit; ++t, ++count)
      target_map[t] = count;
  }

  // copy instructions and store in vector
  std::vector<goto_programt::targett> target_vector;
  target_vector.reserve(target_map.size());

  for (auto t = body_begin; t != loop_exit; ++t)
  {
    goto_programt::targett copied_t = out.add_instruction(*t);
    target_vector.push_back(copied_t);
  }

  // rewrite targets using index-based mapping
  for (unsigned i = 0; i < target_vector.size(); i++)
  {
    goto_programt::targett t = target_vector[i];
    for (auto &target : t->targets)
    {
      std::map<goto_programt::targett, unsigned>::const_iterator m_it =
        target_map.find(target);
      if (m_it != target_map.end()) // intra-loop?
        target = target_vector[m_it->second];
    }
  }
}

/// The condition under which a while or for loop is entered: the negation of
/// its head IF's guard, which ASSUME(entry_cond) stands in for so the head need
/// not be copied. Nil for a do-while, whose head is not a conditional goto but
/// the first instruction of the body -- it always enters (issue #7494).
static expr2tc loop_entry_cond(goto_programt::targett loop_head)
{
  if (loop_head->is_goto() && !loop_head->is_backwards_goto())
    return not2tc(loop_head->guard);
  return expr2tc();
}

/// Where the copied body starts: after the head when ASSUME(entry_cond) stands
/// in for it, at the head otherwise, because a do-while head is a body
/// instruction and skipping it left the verification branch empty (#7494).
static goto_programt::targett
loop_body_begin(goto_programt::targett loop_head, const expr2tc &entry_cond)
{
  return is_nil_expr(entry_cond) ? loop_head : std::next(loop_head);
}

/// Constrain the copied iteration to one another would follow. A do-while's
/// exit test lives on its back edge, so the copy may be the last iteration, and
/// a head invariant need not hold where the loop exits: `x <= 4` on
/// `do x++; while (x < 5)` holds at every head and is false once the exiting
/// iteration leaves x == 5. No-op for the unconditional back edge of a while or
/// for loop (issue #7494).
static void
assume_continue_cond(goto_programt::targett loop_exit, goto_programt &out)
{
  if (!loop_exit->is_goto() || is_true(loop_exit->guard))
    return;

  auto t = out.add_instruction(ASSUME);
  t->guard = loop_exit->guard;
  t->location = loop_exit->location;
  t->location.comment("loop continue condition (verification branch)");
}

void goto_loop_invariant_combinedt::insert_invariant_verification_branch(
  loopst &loop)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // ── 1. Collect invariant expressions ──────────────────────────────────────
  std::vector<expr2tc> invariants =
    extract_invariants_near(loop_head, goto_function.body.instructions.begin());

  if (invariants.empty())
    return;

  // Reuse the legacy side-effect extraction so that any temporaries or
  // function calls used in the invariant are re-evaluated in Branch 1 and
  // when inserting ASSUME(INV) for Branch 2.
  goto_programt side_effects;
  extract_and_remove_side_effects_impl(
    goto_function, loop_head, loop, invariants, side_effects);

  // ── 2. Copy loop body, and the entry condition that stands in for the head
  const expr2tc entry_cond = loop_entry_cond(loop_head);

  goto_programt body_copy;
  copy_loop_body(loop_body_begin(loop_head, entry_cond), loop_exit, body_copy);

  // ── 4. Build Branch 1 ─────────────────────────────────────────────────────
  const auto &loop_vars = loop.get_modified_loop_vars();
  goto_programt branch1;

  // [4a] Base-case ASSERT(INV)
  for (const auto &instr : side_effects.instructions)
    branch1.instructions.push_back(instr);
  for (const auto &inv : invariants)
  {
    auto t = branch1.add_instruction(ASSERT);
    t->guard = inv;
    t->location = loop_head->location;
    t->location.comment("loop invariant base case");
    t->location.property("invariant-base-case");
  }

  // [4b] HAVOC(loop_vars)
  for (const auto &var : loop_vars)
  {
    if (
      config.options.get_bool_option("add-symex-value-sets") &&
      is_pointer_type(var))
      continue;

    auto t = branch1.add_instruction(ASSIGN);
    t->code = code_assign2tc(var, gen_nondet(var->type));
    t->location = loop_head->location;
    t->location.comment("loop invariant havoc (verification branch)");
  }

  // [4c] ASSUME(INV)  — constrain the havoc'd state
  for (const auto &instr : side_effects.instructions)
    branch1.instructions.push_back(instr);
  for (const auto &inv : invariants)
  {
    auto t = branch1.add_instruction(ASSUME);
    t->guard = inv;
    t->location = loop_head->location;
    t->location.comment("loop invariant assume (verification branch)");
  }

  // [4d] ASSUME(entry_cond)  — model being inside the loop
  if (!is_nil_expr(entry_cond) && !is_true(entry_cond))
  {
    auto t = branch1.add_instruction(ASSUME);
    t->guard = entry_cond;
    t->location = loop_head->location;
    t->location.comment("loop entry condition (verification branch)");
  }

  // [4e] One iteration of the loop body
  branch1.destructive_insert(branch1.instructions.end(), body_copy);

  // [4e'] Only an iteration another would follow has to preserve the invariant.
  // PR #3777 describes this for the copied body; it never bit because a
  // do-while's body was not copied at all (issue #7494).
  assume_continue_cond(loop_exit, branch1);

  // [4f] Inductive-step ASSERT(INV)
  for (const auto &instr : side_effects.instructions)
    branch1.instructions.push_back(instr);
  for (const auto &inv : invariants)
  {
    auto t = branch1.add_instruction(ASSERT);
    t->guard = inv;
    t->location = loop_exit->location;
    t->location.comment("loop invariant inductive step");
    t->location.property("invariant-inductive-step");
  }

  // [4g] ASSUME(false) — terminate Branch 1; never falls through
  {
    auto t = branch1.add_instruction(ASSUME);
    t->guard = gen_false_expr();
    t->location = loop_exit->location;
    t->location.comment("end invariant verification branch");
  }

  // ── 5. Prepend the nondet gate ────────────────────────────────────────────
  // "IF !nondet_bool() GOTO loop_head"  →  skip Branch 1 on false branch
  {
    goto_programt gate;
    auto t = gate.add_instruction(GOTO);
    t->guard = not2tc(gen_nondet(get_bool_type()));
    t->targets.push_back(loop_head); // jump to original loop head
    t->location = loop_head->location;

    // Splice the gate at the front of branch1
    branch1.destructive_insert(branch1.instructions.begin(), gate);
  }

  // ── 6. Add ASSUME(INV) at the loop head (Branch 2) ───────────────────────
  // The invariant holds where it is annotated, at the head of an iteration, so
  // that is where k-induction gets to assume it. For a while or for loop the
  // head is the guard, and the assumption goes just inside it, where the body
  // begins. A do-while head is already the first body instruction: advancing
  // past it puts the ASSUME at the body tail, ahead of the loop's own exit
  // test, where a head invariant need not hold -- the exiting iteration then
  // contradicts it, the exit edge is assumed away and every claim after the
  // loop silently disappears (issue #7494).

  // ── 7. Insert before loop_head using splice (NOT insert_swap) ─────────────
  // splice() inserts before loop_head without moving it, so any existing
  // backward-GOTO target that points to loop_head continues to do so.
  goto_function.body.destructive_insert(loop_head, branch1);

  for (const auto &inv : invariants)
  {
    goto_programt assume_inv;
    for (const auto &instr : side_effects.instructions)
      assume_inv.instructions.push_back(instr);
    auto t = assume_inv.add_instruction(ASSUME);
    t->guard = inv;
    t->location = loop_head->location;
    t->location.comment("loop invariant assume (k-induction hint)");
    if (!is_nil_expr(entry_cond))
      loop_head++;
    goto_function.body.insert_swap(loop_head, assume_inv);
  }
}
