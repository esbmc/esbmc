#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_skip.h>
#include <set>
#include <stack>

void remove_unreachable(goto_functionst &goto_functions)
{
  for(auto &fun_it : goto_functions.function_map)
    remove_unreachable(fun_it.second);
}

void remove_unreachable(goto_functiont &goto_function)
{
  remove_unreachable(goto_function.body);
}

void remove_unreachable(goto_programt &goto_program)
{
  std::set<goto_programt::const_targett> reachable;
  std::stack<goto_programt::const_targett> working;

  working.push(goto_program.instructions.begin());

  while(!working.empty())
  {
    goto_programt::const_targett t = working.top();
    working.pop();

    if(
      reachable.find(t) == reachable.end() &&
      t != goto_program.instructions.end())
    {
      reachable.insert(t);
      goto_programt::const_targetst successors;
      goto_program.get_successors(t, successors);

      for(goto_programt::const_targetst::const_iterator s_it =
            successors.begin();
          s_it != successors.end();
          s_it++)
        working.push(*s_it);
    }
  }

  // All unreachable code apart from END_FUNCTION instructions
  // is transformed into SKIP
  Forall_goto_program_instructions(it, goto_program)
  {
    if(reachable.find(it) == reachable.end() && !it->is_end_function())
      it->make_skip();
  }

  // Finally, all introduced SKIPs are now removed
  remove_skip(goto_program);
}
