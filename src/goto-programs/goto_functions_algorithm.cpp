#include <goto-programs/goto_functions_algorithm.h>
#include <goto-programs/goto_loops.h>
#include <goto-programs/remove_no_op.h>

bool goto_functions_algorithm::run(goto_functionst &goto_functions)
{
  runOnProgram(goto_functions);
  Forall_goto_functions (it, goto_functions)
  {
    number_of_functions++;
    runOnFunction(*it);
    if (it->second.body_available)
    {
      // runOnFunction may have inserted instructions, and insert_swap does not
      // renumber: fresh nodes keep location_number 0. goto_loopst classifies
      // back edges by comparing location numbers, so it has to see current
      // ones -- the update() below only runs once every function is done.
      it->second.body.update();

      goto_loopst goto_loops(it->first, goto_functions, it->second);
      auto function_loops = goto_loops.get_loops();
      number_of_loops += function_loops.size();
      if (function_loops.size())
      {
        goto_functiont &goto_function = it->second;
        goto_programt &goto_program = goto_function.body;

        // Foreach loop in the function
        for (auto itt = function_loops.rbegin(); itt != function_loops.rend();
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
  std::pair<const irep_idt, goto_functiont> &)
{
  return true;
}
