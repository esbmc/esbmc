#include "irep2/irep2_utils.h"
#include <goto-programs/goto_loops.h>
#include <util/expr_util.h>

bool check_var_name(const expr2tc &expr)
{
  if (!is_symbol2t(expr))
    return false;

  symbol2t s = to_symbol2t(expr);
  std::string identifier = s.thename.as_string();

  std::size_t found = identifier.find("__ESBMC_");
  if (found != std::string::npos)
    return false;

  found = identifier.find("return_value___");
  if (found != std::string::npos)
    return false;

  found = identifier.find("pthread_lib");
  if (found != std::string::npos)
    return false;

  // Don't add variables that we created for k-induction
  found = identifier.find("$");
  if (found != std::string::npos)
    return false;

  if (identifier == "__func__")
    return false;

  if (identifier == "__PRETTY_FUNCTION__")
    return false;

  if (identifier == "__LINE__")
    return false;

  return true;
}

void goto_loopst::find_function_loops()
{
  for (goto_programt::instructionst::iterator it =
         goto_function.body.instructions.begin();
       it != goto_function.body.instructions.end();
       it++)
  {
    // We found a loop, let's record its instructions
    if (it->is_backwards_goto())
    {
      assert(it->targets.size() == 1);
      goto_programt::instructionst::iterator &loop_head = *it->targets.begin();
      goto_programt::instructionst::iterator &loop_exit = it;

      // This means something like:
      // A: goto A;
      if (loop_head->location_number == loop_exit->location_number)
      {
        // In TACAS, this is a common setup for reaching a "dead state" (no violation)
        it->make_assumption(gen_false_expr());
        continue;
      }
      create_function_loop(loop_head, loop_exit);
    }
  }
}

void goto_loopst::create_function_loop(
  goto_programt::instructionst::iterator loop_head,
  goto_programt::instructionst::iterator loop_exit)
{
  goto_programt::instructionst::iterator it = loop_head;

  function_loops.push_front(loopst());
  function_loopst::iterator it1 = function_loops.begin();

  // Set original iterators
  it1->set_original_loop_head(loop_head);
  it1->set_original_loop_exit(loop_exit);

  // Push the current function name to the list of functions
  std::vector<irep_idt> function_names;
  function_names.push_back(function_name);

  // Copy the loop body
  std::size_t size = 0;
  while (it != loop_exit)
  {
    // This should be done only when we're running k-induction
    // Maybe a flag on the class?
    get_modified_variables(it, it1, function_names);
    ++it;

    // Count the number of instruction
    ++size;
  }

  // Include loop_exit
  it1->set_size(size + 1);
}

void goto_loopst::get_modified_variables(
  goto_programt::instructionst::iterator instruction,
  function_loopst::iterator loop,
  std::vector<irep_idt> &function_names)
{
  if (instruction->is_assign())
  {
    const code_assign2t &assign = to_code_assign2t(instruction->code);
    add_loop_var(*loop, assign.target, true);
  }
  else if (instruction->is_function_call())
  {
    // Functions are a bit tricky
    code_function_call2t &function_call =
      to_code_function_call2t(instruction->code);

    // Don't do function pointers
    if (is_dereference2t(function_call.function))
      return;

    // First, add its return
    add_loop_var(*loop, function_call.ret, true);

    // The run over the function body and get the modified variables there
    irep_idt &identifier = to_symbol2t(function_call.function).thename;

    // This means recursion, do nothing
    if (
      std::find(function_names.begin(), function_names.end(), identifier) !=
      function_names.end())
      return;

    // We didn't entered this function yet, so add it to the list
    function_names.push_back(identifier);

    // find code in function map
    goto_functionst::function_mapt::iterator it =
      goto_functions.function_map.find(identifier);

    if (it == goto_functions.function_map.end())
    {
      log_error("failed to find `{}' in function_map", id2string(identifier));
      abort();
    }

    // Avoid iterating over functions that don't have a body
    if (!it->second.body_available)
      return;

    for (goto_programt::instructionst::iterator head =
           it->second.body.instructions.begin();
         head != it->second.body.instructions.end();
         ++head)
    {
      get_modified_variables(head, loop, function_names);
    }
  }
  else if (
    instruction->is_goto() || instruction->is_assert() ||
    instruction->is_assume())
  {
    add_loop_var(*loop, instruction->guard, false);
  }
  else if (instruction->is_end_function())
  {
    function_names.pop_back();
  }
}

void goto_loopst::add_loop_var(
  loopst &loop,
  const expr2tc &expr,
  bool is_modified)
{
  if (is_nil_expr(expr))
    return;

  expr->foreach_operand([this, &loop, &is_modified](const expr2tc &e) {
    add_loop_var(loop, e, is_modified);
  });

  if (is_symbol2t(expr) && check_var_name(expr))
  {
    if (is_modified)
      loop.add_modified_var_to_loop(expr);
    else
      loop.add_unmodified_var_to_loop(expr);
  }
}

void goto_loopst::dump() const
{
  for (auto &function_loop : function_loops)
    function_loop.dump();
}
