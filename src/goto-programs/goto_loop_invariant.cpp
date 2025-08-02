
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
 *     assume (!(lb <= ub));
 *     assume (lb - 1 <= ub && ub < size && mid == ((unsigned int)lb + (unsigned int)ub) >> 1);
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

  // 1. Insert ASSERT invariant before loop (base case)
  insert_assert_before_loop(loop_head, invariants);

  // // 2. Insert HAVOC and ASSUME before loop condition (after base case assert)
   insert_havoc_and_assume_before_condition(loop_head, loop, invariants);

  // // 3. Insert inductive step verification and loop termination
   insert_inductive_step_and_termination(loop, invariants);
}

std::vector<expr2tc> goto_loop_invariantt::extract_loop_invariants(const loopst &loop)
{
  std::vector<expr2tc> invariants;

  goto_programt::targett loop_head = loop.get_original_loop_head();

  // Safety check: ensure the dest is not empty
  if (loop_head == goto_function.body.instructions.begin())
    return invariants;

  // Check immediately preceding instruction
  goto_programt::targett prev_it = loop_head;
  --prev_it;

  if (prev_it->is_loop_invariant())
  {

    auto const& current_invariants = prev_it->get_loop_invariants();

    if (current_invariants.size() == 1)
    {
      // add single invariant
      invariants.push_back(current_invariants.front());
    }
    else if (current_invariants.size() > 1)
    {
      // Combine to one && format invariant
      auto it = current_invariants.begin();
      auto combined_invariant = *it;  // first element
      ++it;  // move to second element
      
      for (; it != current_invariants.end(); ++it)
      {
        combined_invariant = and2tc(combined_invariant, *it);
      }
      
      // return one combined invariant
      invariants.push_back(combined_invariant);
    }
    // if current_invariants.empty(), do nothing, return empty invariants
  }
  
  return invariants;
}

void goto_loop_invariantt::insert_assert_before_loop(
  goto_programt::targett &loop_head,
  const std::vector<expr2tc> &invariants)
{
  goto_programt dest;
  
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
  const std::vector<expr2tc> &invariants)
{
  // Find the loop condition (IF instruction) - this should be right at loop_head
  goto_programt::targett condition_it = loop_head;
  while (condition_it != goto_function.body.instructions.end() && !condition_it->is_goto())
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
  
  // 2. Insert ASSUME with invariants only (we'll assume we're entering the loop)
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
  const std::vector<expr2tc> &invariants)
{
  // Find the end of loop body (before GOTO back to loop head)
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  goto_programt::targett insert_point = loop_exit;
  
  // Move back to find the last assignment in loop body
  // Add safety check to prevent going beyond beginning
  size_t safety_counter = 0;
  const size_t max_iterations = 10000; // Prevent infinite loop of searching. I did not provide a option now.
  
  while (insert_point != goto_function.body.instructions.begin() && 
         !insert_point->is_assign() && 
         safety_counter < max_iterations)
  {
    --insert_point;
    ++safety_counter;
  }
  
  if (insert_point == goto_function.body.instructions.begin() || 
      safety_counter >= max_iterations)
  {
    // No assignment found in loop body or too many iterations
    // Insert at the beginning of loop body as fallback
    insert_point = loop.get_original_loop_head();
    ++insert_point;
  }
  else
  {
    // Move to AFTER the last assignment
    ++insert_point;
  }
  
  goto_programt dest;
  
  // 1. Insert ASSERT for inductive step verification
  for (const auto &invariant : invariants)
  {
    // Create assert instruction for each invariant
    goto_programt::targett t = dest.add_instruction(ASSERT);
    t->guard = invariant;
    t->location = loop_exit->location;
    t->location.comment("loop invariant inductive step");
  }
  
  // 2. Insert ASSUME(FALSE) to terminate the loop
  goto_programt::targett t = dest.add_instruction(ASSUME);
  t->guard = gen_false_expr();
  t->location = loop_exit->location;
  t->location.comment("loop termination");
  
  // Insert at the insert point
  goto_function.body.insert_swap(insert_point, dest);
} 