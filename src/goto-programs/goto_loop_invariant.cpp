
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
 */

#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/remove_no_op.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>
#include <util/options.h>
#include <irep2/irep2_utils.h>

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

  while (it != begin && dist < kMaxInvariantSearchBack)
  {
    --it;
    ++dist;

    if (!it->is_loop_invariant())
      continue;

    const std::list<expr2tc> &inv_list = it->get_loop_invariants();
    if (inv_list.empty())
      continue;

    if (inv_list.size() == 1)
    {
      invariants.push_back(inv_list.front());
    }
    else
    {
      auto jt = inv_list.begin();
      expr2tc combined = *jt;
      for (++jt; jt != inv_list.end(); ++jt)
        combined = and2tc(combined, *jt);
      invariants.push_back(combined);
    }
    break;
  }

  return invariants;
}

void goto_loop_invariant(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_loop_invariantt(it->first, goto_functions, it->second);

  goto_functions.update();
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
    return; // No invariants found, skip this loop

  // Extract invariant-related DECL/FUNCTION_CALLs and re-insert before each
  // ASSERT/ASSUME.
  goto_programt side_effects;
  extract_and_remove_side_effects(loop_head, loop, invariants, side_effects);

  // 1. Insert ASSERT invariant before loop (base case)
  insert_assert_before_loop(loop_head, invariants, side_effects);

  // 2. Insert HAVOC and ASSUME before loop condition (after base case assert)
  insert_havoc_and_assume_before_condition(
    loop_head, loop, invariants, side_effects);

  // 3. Insert inductive step verification and loop termination
  insert_inductive_step_and_termination(loop, invariants, side_effects);
}

std::vector<expr2tc>
goto_loop_invariantt::extract_loop_invariants(const loopst &loop)
{
  return extract_invariants_near(
    loop.get_original_loop_head(),
    goto_function.body.instructions.begin());
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

  for (unsigned i = 0; i < expr->get_num_sub_exprs(); ++i)
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
    is_sideeffect2t(expr) && to_sideeffect2t(expr).kind == sideeffect2t::nondet)
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

void goto_loop_invariantt::extract_and_remove_side_effects(
  goto_programt::targett loop_head,
  const loopst &loop,
  const std::vector<expr2tc> &invariants,
  goto_programt &side_effects_out)
{
  // Collect symbol names referenced in the invariants.
  std::set<irep_idt> inv_symbols;
  for (const auto &inv : invariants)
    collect_symbols(inv, inv_symbols);

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
      if (inv_symbols.count(lhs_sym) && !is_likely_compiler_temp(lhs_sym))
        break;
      if (is_trivial_rhs(assign.source))
        continue;
      to_remove.push_back(search);
      collect_symbols(assign.source, dep_symbols);
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

    break;
  }

  if (to_remove.empty())
    return;

  for (auto rit = to_remove.rbegin(); rit != to_remove.rend(); ++rit)
  {
    side_effects_out.instructions.push_back(**rit);
    goto_function.body.instructions.erase(*rit);
  }
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
  }

  // Insert before the loop head
  goto_function.body.insert_swap(loop_head, dest);
}

void goto_loop_invariantt::insert_havoc_and_assume_before_condition(
  goto_programt::targett &loop_head,
  const loopst &loop,
  const std::vector<expr2tc> &invariants,
  const goto_programt &side_effects)
{
  // Find the loop condition (IF instruction) - this should be right at loop_head
  goto_programt::targett condition_it = loop_head;
  while (condition_it != goto_function.body.instructions.end() &&
         !condition_it->is_goto())
    ++condition_it;

  if (condition_it == goto_function.body.instructions.end())
    return; // No loop condition found

  // Insert BEFORE the loop condition (after the base case assert)
  goto_programt::targett insert_point = condition_it;

  goto_programt dest;

  // 1. Insert HAVOC (nondet assignments) before the loop condition
  auto const &loop_vars = loop.get_modified_loop_vars();
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
  }

  // 2. Emit side-effect instructions (use havoc'd variables).
  for (const auto &instr : side_effects.instructions)
    dest.instructions.push_back(instr);

  // 3. ASSUME invariants (entering the loop).
  for (const auto &invariant : invariants)
  {
    // Create assume instruction: just the invariant
    goto_programt::targett t = dest.add_instruction(ASSUME);
    t->guard = invariant;
    t->location = loop_head->location;
    t->location.comment("loop invariant step case");
  }

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

  // 2. ASSERT for inductive step.
  for (const auto &invariant : invariants)
  {
    // Create assert instruction for each invariant
    goto_programt::targett t = dest.add_instruction(ASSERT);
    t->guard = invariant;
    t->location = loop_exit->location;
    t->location.comment("loop invariant inductive step");
  }

  // 3. Insert ASSUME(FALSE) to terminate the loop
  goto_programt::targett t = dest.add_instruction(ASSUME);
  t->guard = gen_false_expr();
  t->location = loop_exit->location;
  t->location.comment("loop termination");

  // Insert at the insert point
  goto_function.body.insert_swap(insert_point, dest);
}

// ---------------------------------------------------------------------------
// Combined mode implementation
// ---------------------------------------------------------------------------

void goto_loop_invariant_combined(goto_functionst &goto_functions)
{
  Forall_goto_functions(it, goto_functions)
    if (it->second.body_available)
      goto_loop_invariant_combinedt(it->first, goto_functions, it->second);

  goto_functions.update();
}

void goto_loop_invariant_combinedt::process_loops_combined()
{
  for (auto &loop : function_loops)
    insert_invariant_verification_branch(loop);
}

bool goto_loop_invariant_combinedt::copy_loop_body(
  goto_programt::targett loop_head,
  goto_programt::targett loop_exit,
  goto_programt &out) const
{
  // Build the set of location numbers that belong to the loop body
  // [loop_head, loop_exit] so we can detect outward-jumping GOTOs.
  std::unordered_set<unsigned> body_locs;
  for (auto it = loop_head; it != std::next(loop_exit); ++it)
    if (it->location_number != 0)
      body_locs.insert(it->location_number);

  // Copy body instructions: (loop_head, loop_exit)  — exclusive on both ends.
  for (auto it = std::next(loop_head); it != loop_exit; ++it)
  {
    // Reject loops that jump out of their own body via a forward GOTO.
    if (it->is_goto() && !it->is_backwards_goto())
    {
      for (const auto &tgt : it->targets)
      {
        unsigned tgt_loc = tgt->location_number;
        if (tgt_loc != 0 &&
            body_locs.find(tgt_loc) == body_locs.end())
          return false; // complex loop — skip Branch 1
      }
    }

    out.instructions.push_back(*it);
  }

  return true;
}

void goto_loop_invariant_combinedt::insert_invariant_verification_branch(
  loopst &loop)
{
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // ── 1. Collect invariant expressions ──────────────────────────────────────
  std::vector<expr2tc> invariants = extract_invariants_near(
    loop_head, goto_function.body.instructions.begin());

  if (invariants.empty())
    return;

  // ── 2. Copy loop body (fail gracefully for complex loops) ─────────────────
  goto_programt body_copy;
  if (!copy_loop_body(loop_head, loop_exit, body_copy))
    return; // Fall back to ASSUME-only mode in goto_k_induction

  // ── 3. Extract the loop entry condition ───────────────────────────────────
  // loop_head is "IF !(cond) GOTO exit", so the entry condition is "cond"
  // i.e. the negation of the GOTO guard.
  expr2tc entry_cond;
  if (loop_head->is_goto() && !loop_head->is_backwards_goto())
    entry_cond = not2tc(loop_head->guard);
  // If loop_head is not a conditional GOTO (e.g. do-while), entry_cond
  // stays nil and we omit the ASSUME(entry_cond) — always enters.

  // ── 4. Build Branch 1 ─────────────────────────────────────────────────────
  const auto &loop_vars = loop.get_modified_loop_vars();
  goto_programt branch1;

  // [4a] Base-case ASSERT(INV)
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
    if (config.options.get_bool_option("add-symex-value-sets") &&
        is_pointer_type(var))
      continue;

    auto t = branch1.add_instruction(ASSIGN);
    t->code = code_assign2tc(var, gen_nondet(var->type));
    t->location = loop_head->location;
    t->location.comment("loop invariant havoc (verification branch)");
  }

  // [4c] ASSUME(INV)  — constrain the havoc'd state
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
  branch1.instructions.splice(
    branch1.instructions.end(), body_copy.instructions);

  // [4f] Inductive-step ASSERT(INV)
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
    branch1.instructions.splice(branch1.instructions.begin(), gate.instructions);
  }

  // ── 6. Insert before loop_head using splice (NOT insert_swap) ─────────────
  // splice() inserts before loop_head without moving it, so any existing
  // backward-GOTO target that points to loop_head continues to do so.
  goto_function.body.instructions.splice(
    loop_head, branch1.instructions);

  // ── 7. Add ASSUME(INV) at end of original loop body (Branch 2) ───────────
  // Inserting ASSUME(INV) just before loop_exit (the backward GOTO) means
  // k-induction will see the assumption at the end of every iteration without
  // any modification to goto_k_induction itself.
  for (const auto &inv : invariants)
  {
    goto_programt assume_inv;
    auto t = assume_inv.add_instruction(ASSUME);
    t->guard = inv;
    t->location = loop_exit->location;
    t->location.comment("loop invariant assume (k-induction hint)");
    goto_function.body.instructions.splice(loop_exit, assume_inv.instructions);
  }
}
