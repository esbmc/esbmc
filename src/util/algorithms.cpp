#include <util/algorithms.h>
#include <util/message.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_skip.h>
bool goto_functions_algorithm::run(goto_functionst &goto_functions)
{
  Forall_goto_functions(it, goto_functions)
  {
    number_of_functions++;
    runOnFunction(*it);
    if(it->second.body_available)
    {
      goto_loopst goto_loops(it->first, goto_functions, it->second);
      auto function_loops = goto_loops.get_loops();
      number_of_loops += function_loops.size();
      if(function_loops.size())
      {
        goto_functiont &goto_function = it->second;
        goto_programt &goto_program = goto_function.body;

        // Foreach loop in the function
        for(auto itt = function_loops.rbegin(); itt != function_loops.rend();
            ++itt)
          runOnLoop(*itt, goto_program);
      }
    }
  }
  goto_functions.update();
  return true;
}

bool goto_functions_algorithm::runOnLoop(loopst &, goto_programt &)
{
  return true;
}

bool goto_functions_algorithm::runOnFunction(
  std::pair<const dstring, goto_functiont> &)
{
  return true;
}
