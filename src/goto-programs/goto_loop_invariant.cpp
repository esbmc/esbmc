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
  goto_programt::targett loop_exit = loop.get_original_loop_exit();

  // Extract loop invariants from LOOP_INVARIANT instructions
  std::vector<expr2tc> invariants = extract_loop_invariants(loop);

  if (invariants.empty())
    return; // No invariants found, skip this loop

  // 1. Insert ASSERT invariant before loop (base case)
  insert_assert_before_loop(loop_head, invariants);

  // 2. Insert HAVOC and ASSUME before loop condition (after base case assert)
  insert_havoc_and_assume_before_condition(loop_head, loop, invariants);

  // 3. Insert ASSERT invariant at loop body end (inductive step)
  // insert_assert_at_loop_end(loop_exit, invariants);
}

std::vector<expr2tc> goto_loop_invariantt::extract_loop_invariants(const loopst &loop)
{
  std::vector<expr2tc> invariants;
  
  // Walk through the loop body to find LOOP_INVARIANT instructions
  goto_programt::targett loop_head = loop.get_original_loop_head();
  goto_programt::targett loop_exit = loop.get_original_loop_exit();
  
  for (goto_programt::targett it = loop_head; it != loop_exit; ++it)
  {
    if (it->is_loop_invariant())
    {
      // Extract all invariants from this instruction
      for (const auto &invariant : it->get_loop_invariants())
      {
        invariants.push_back(invariant);
      }
    }
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
  
  // 3. Insert inductive step verification and loop termination
  insert_inductive_step_and_termination(loop, invariants);
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
  int safety_counter = 0;
  const int max_iterations = 1000; // Prevent infinite loop
  
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