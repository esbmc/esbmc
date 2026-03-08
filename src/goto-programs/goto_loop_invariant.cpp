
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
  extract_and_remove_side_effects(loop_head, invariants, side_effects);

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
  std::vector<expr2tc> invariants;

  goto_programt::targett loop_head = loop.get_original_loop_head();

  // Cannot search backwards past the very first instruction.
  if (loop_head == goto_function.body.instructions.begin())
    return invariants;

  // Search backwards from loop head to find the LOOP_INVARIANT instruction.
  // We stop at the first one found (the invariant for THIS loop), since the
  // user must place __ESBMC_loop_invariant immediately before its loop.
  goto_programt::targett search_it = loop_head;

  size_t search_distance = 0;

  while (search_it != goto_function.body.instructions.begin() &&
         search_distance < kMaxInvariantSearchBack)
  {
    --search_it;
    ++search_distance;

    if (search_it->is_loop_invariant())
    {
      auto const &current_invariants = search_it->get_loop_invariants();

      if (current_invariants.size() == 1)
      {
        invariants.push_back(current_invariants.front());
        break;
      }
      else if (current_invariants.size() > 1)
      {
        // Multiple sub-expressions: fold them into a single && conjunction.
        auto it = current_invariants.begin();
        auto combined = *it;
        for (++it; it != current_invariants.end(); ++it)
          combined = and2tc(combined, *it);
        invariants.push_back(combined);
        break;
      }
      // current_invariants.empty(): skip and keep searching backwards.
    }
  }

  return invariants;
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

void goto_loop_invariantt::extract_and_remove_side_effects(
  goto_programt::targett loop_head,
  const std::vector<expr2tc> &invariants,
  goto_programt &side_effects_out)
{
  // Collect symbol names referenced in the invariants.
  std::set<irep_idt> inv_symbols;
  for (const auto &inv : invariants)
    collect_symbols(inv, inv_symbols);

  if (inv_symbols.empty())
    return;

  // Find the LOOP_INVARIANT instruction preceding loop_head.
  // Uses the same search limit as extract_loop_invariants so the two
  // backward passes are consistent.
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
    return; // no LOOP_INVARIANT found

  // Walk backwards from the LOOP_INVARIANT: collect DECL/FUNCTION_CALL
  // instructions that define temporaries used in the invariant; stop at the
  // first unrecognised instruction.
  std::vector<goto_programt::targett> to_remove;
  std::set<irep_idt> fc_return_syms;

  goto_programt::targett search = loop_inv_it;
  while (search != goto_function.body.instructions.begin())
  {
    --search;

    if (search->is_function_call())
    {
      if (!is_code_function_call2t(search->code))
        break;
      const code_function_call2t &fc = to_code_function_call2t(search->code);
      if (!is_nil_expr(fc.ret) && is_symbol2t(fc.ret))
      {
        irep_idt sym = to_symbol2t(fc.ret).get_symbol_name();
        if (inv_symbols.count(sym))
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
      if (fc_return_syms.count(decl.value))
      {
        to_remove.push_back(search);
        continue;
      }
      break;
    }

    // Any other instruction (ASSIGN, GOTO, SKIP, ASSUME, etc.) ends the
    // backward search so we only collect the contiguous DECL/FC block.
    break;
  }

  if (to_remove.empty())
    return;

  // Move collected instructions into side_effects_out (original order) and
  // erase from the GOTO program.
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
