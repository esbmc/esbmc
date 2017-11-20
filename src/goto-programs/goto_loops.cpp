/*
 * loopst.cpp
 *
 *  Created on: Jun 30, 2015
 *      Author: mramalho
 */

#include <goto-programs/goto_loops.h>
#include <util/expr_util.h>

bool check_var_name(const expr2tc &expr)
{
  if(!is_symbol2t(expr))
    return false;

  symbol2t s = to_symbol2t(expr);
  std::string identifier = s.thename.as_string();

  std::size_t found = identifier.find("__ESBMC_");
  if(found != std::string::npos)
    return false;

  found = identifier.find("return_value___");
  if(found != std::string::npos)
    return false;

  found = identifier.find("pthread_lib");
  if(found != std::string::npos)
    return false;

  // Don't add variables that we created for k-induction
  found = identifier.find("$");
  if(found != std::string::npos)
    return false;

  if(identifier == "__func__")
    return false;

  if(identifier == "__PRETTY_FUNCTION__")
    return false;

  if(identifier == "__LINE__")
    return false;

  return true;
}

void goto_loopst::find_function_loops()
{
  for(goto_programt::instructionst::iterator
      it=goto_function.body.instructions.begin();
      it!=goto_function.body.instructions.end();
      it++)
  {
    // We found a loop, let's record its instructions
    if (it->is_backwards_goto())
    {
      assert(it->targets.size() == 1);
      create_function_loop(*it->targets.begin(), it);
    }
  }
}

void goto_loopst::create_function_loop(
  goto_programt::instructionst::iterator loop_head,
  goto_programt::instructionst::iterator loop_exit)
{
  // This means something like:
  // A: goto A;
  // There is no body, so we can skip it
  if(loop_head->location_number == loop_exit->location_number)
    return;

  goto_programt::instructionst::iterator it=loop_head;

  function_loops.push_front(loopst(context));
  function_loopst::iterator it1 = function_loops.begin();

  // Set original iterators
  it1->set_original_loop_head(loop_head);
  it1->set_original_loop_exit(loop_exit);

  std::size_t size = 0;
  // Copy the loop body
  while (it != loop_exit)
  {
    // This should be done only when we're running k-induction
    // Maybe a flag on the class?
    get_modified_variables(it, it1, function_name);
    ++it;

    // Count the number of instruction
    ++size;
  }

  // Include loop_exit
  it1->set_size(size+1);
}

void goto_loopst::get_modified_variables(
  goto_programt::instructionst::iterator instruction,
  function_loopst::iterator loop,
  const irep_idt &_function_name)
{
  if(instruction->is_assign())
  {
    const code_assign2t &assign = to_code_assign2t(instruction->code);
    add_loop_var(*loop, assign.target);
  }
  else if(instruction->is_function_call())
  {
    // Functions are a bit tricky
    code_function_call2t &function_call =
      to_code_function_call2t(instruction->code);

    // Don't do function pointers
    if(is_dereference2t(function_call.function))
      return;

    // First, add its return
    add_loop_var(*loop, function_call.ret);

    // The run over the function body and get the modified variables there
    irep_idt &identifier = to_symbol2t(function_call.function).thename;

    // This means recursion, do nothing
    if(identifier == _function_name)
      return;

    // find code in function map
    goto_functionst::function_mapt::iterator it =
      goto_functions.function_map.find(identifier);

    if (it == goto_functions.function_map.end()) {
      std::cerr << "failed to find `" + id2string(identifier) +
                   "' in function_map";
      abort();
    }

    // Avoid iterating over functions that don't have a body
    if(!it->second.body_available)
      return;

    for(goto_programt::instructionst::iterator head=
        it->second.body.instructions.begin();
        head != it->second.body.instructions.end();
        ++head)
    {
      get_modified_variables(head, loop, identifier);
    }
  }
}

void goto_loopst::add_loop_var(loopst &loop, const expr2tc& expr)
{
  if(is_nil_expr(expr))
    return;

  expr->foreach_operand([this, &loop] (const expr2tc &e) {
    add_loop_var(loop, e);
  });

  if(is_symbol2t(expr) && check_var_name(expr))
    loop.add_var_to_loop(expr);
}

void goto_loopst::dump()
{
  for(auto & function_loop : function_loops)
    function_loop.dump();
}
