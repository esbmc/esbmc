
/*
 * This function is used to check loop invariants.
 * It takes phases:
 * 1. Check if the invariants are satisfiable before entering the loop (Step 1)
 * 2. Capture all related variables in the loop (not done yet), then set all the variables to nondet
 * 3. Set the loop invariant as the assumption (Step k)
 * 4. Enter the loop (only run a single step of the loop)
 * 5. Check if the invariant is satisfiable after the loop (Step k+1)
 * 6. If the invariant is not satisfied, the function will return a warning (loop invariants are not preserved).
 * 7. If the invariant is satisfied, the function will return true.
 * 8. Use the loop invariants as assumptions in the following steps.
 * 
 * For example:
 * int binary_search(int *arr, int size, int target) {
 *   int lb = 0;
 *   int ub = size - 1;
 *   while (lb <= ub) {
 *     int mid = (lb + ub) / 2;
 *     if (arr[mid] == target) {
 *       return mid;
 *     }
 *     if (arr[mid] < target) {
 *       left = mid + 1;
 *     } else {
 *       right = mid - 1;
 *     }
 *   }
 *   return -1;
 * }
 * 
 * We can find the loop invariant as: lb - 1 <= ub && ub < size && mid == ((unsigned int)lb + (unsigned int)ub) >> 1
 * 
 * We can use the following steps to check the loop invariant:
 * 
 * int binary_search(int *arr, int size, int target) {
 *   int lb = 0;
 *   int ub = size - 1;
 *   // 1. Assert invariants before entering the loop
 *   assert(lb -1 <= ub && ub < size && mid == ((unsigned int)lb + (unsigned int)ub) >> 1);
 * 
 *   // 2. Capture all related variables in the loop
 *   int mid = nondet_int();
 *   int lb = nondet_int();
 *   int ub = nondet_int();
 * 
 *   // 3. Set the loop invariant as the assumption
 *   assume(lb - 1 <= ub && ub < size && mid == ((unsigned int)lb + (unsigned int)ub) >> 1); // Step k
 * 
 *   // 4. Enter the loop (only run a single step of the loop)
 *   // Branch 1: 
 *   if (lb <= ub) {
 *     int mid = (lb + ub) / 2;
 *     if (arr[mid] == target) {
 *       return mid;
 *     }
 *     if (arr[mid] < target) {
 *       lb = mid + 1;
 *     } else {
 *       ub = mid - 1;
 *     }
 * 
 *     // 5. Check if the invariant is satisfiable after the loop // Step k
 *     assert(lb - 1 <= ub && ub < size && mid == ((unsigned int)lb + (unsigned int)ub) >> 1); //should use Multi-property for these assertion analysis
 * 
 *     // 6. Terminate the loop
 *     assume (false);
 *   }
 *   return -1;
 *   
 * // Branch 2: 
 *   else {
 *     // invariant + !condition (need to add theloop exit condition)
 *     // we did not extra explicitly add the loop exit condition, it is from the previous step
 *     //assume (!(lb <= ub)); from if-else branch
 *     //assume (lb - 1 <= ub && ub < size && mid == ((unsigned int)lb + (unsigned int)ub) >> 1); from the previous assumption
 *     // we assume the invariant is preserved after the loop, then we can continue on this branch to check the following assertions.
 *   }
 * }
 * 
 * 
 * 
 */

#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/remove_no_op.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/std_expr.h>
#include <util/options.h>
#include <irep2/irep2_utils.h>

void goto_loop_invariant(goto_functionst &goto_functions)
{
  Forall_goto_functions (it, goto_functions)
    if (it->second.body_available)
      goto_loop_invariantt(it->first, goto_functions, it->second);

  goto_functions.update();
}

void goto_loop_invariantt::goto_loop_invariant()
{
  // Process each loop in the function
  for (auto &function_loop : function_loops)
  {
    if (function_loop.get_modified_loop_vars().empty())
      continue;

    // Start the loop conversion with invariant instrumentation
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
  std::vector<expr2tc> invariants;

  goto_programt::targett loop_head = loop.get_original_loop_head();

  // Safety check: ensure the dest is not empty
  if (loop_head == goto_function.body.instructions.begin())
    return invariants;

  // because we have already combined the loop invariant into one, we should be easy to say it is safe to break.
  // Search backwards from loop head to find LOOP_INVARIANT
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

      // Simple approach: take the first LOOP_INVARIANT found before this loop
      // This works because LOOP_INVARIANT should be placed immediately before its loop
      if (current_invariants.size() == 1)
      {
        // add single invariant
        invariants.push_back(current_invariants.front());
        break; // Found the invariant for this loop, stop searching
      }
      else if (current_invariants.size() > 1)
      {
        // Combine to one && format invariant
        auto it = current_invariants.begin();
        auto combined_invariant = *it; // first element
        ++it;                          // move to second element

        for (; it != current_invariants.end(); ++it)
        {
          combined_invariant = and2tc(combined_invariant, *it);
        }

        // return one combined invariant
        invariants.push_back(combined_invariant);
        break; // Found the invariant for this loop, stop searching
      }
      // if current_invariants.empty(), continue searching
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
  if (is_sideeffect2t(expr) && to_sideeffect2t(expr).kind == sideeffect2t::nondet)
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
          if (!inv_symbols.count(sym) &&
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
