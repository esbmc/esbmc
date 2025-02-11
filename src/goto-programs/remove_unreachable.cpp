#include <goto-programs/remove_unreachable.h>
#include <set>
#include <stack>
#include <prefix.h>

// This method iterates through all GOTO functions in the program
// and removes all unreachable instructions from each of them.
// (Note that we do not attempt to detect and remove unused functions yet.)
void remove_unreachable(goto_functionst &goto_functions)
{
  for (auto &fun_it : goto_functions.function_map)
    remove_unreachable(fun_it.second);
}

// This removes all unreachable instructions from the body
// (i.e., GOTO program) of the given GOTO function.
void remove_unreachable(goto_functiont &goto_function)
{
  remove_unreachable(goto_function.body);
}

// This removes all unreachable instructions in the given GOTO
// program. An instruction is deemed to be reachable if it is
// a successor of another instruction.
void remove_unreachable(goto_programt &goto_program)
{
  std::set<goto_programt::const_targett> reachable;
  std::stack<goto_programt::const_targett> working;

  working.push(goto_program.instructions.begin());

  // Building a set of reachable instructions.
  // Note that the first instruction in the GOTO program is considered to be
  // always reachable.
  while (!working.empty())
  {
    goto_programt::const_targett t = working.top();
    working.pop();

    if (
      reachable.find(t) == reachable.end() &&
      t != goto_program.instructions.end())
    {
      reachable.insert(t);
      goto_programt::const_targetst successors;
      goto_program.get_successors(t, successors);

      for (goto_programt::const_targetst::const_iterator s_it =
             successors.begin();
           s_it != successors.end();
           s_it++)
        working.push(*s_it);
    }
  }

  // Removing all the instructions that do not appear in the constructed
  // reachable set. Note that we never remove the END_FUNCTION instructions
  // (even when they are unreachable)
  auto it = goto_program.instructions.begin();
  while (it != goto_program.instructions.end())
  {
    if (reachable.find(it) == reachable.end() && !it->is_end_function())
      it = goto_program.instructions.erase(it);
    else
      ++it;
  }
}
