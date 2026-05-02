#include <goto-programs/goto_reassociate.h>

#include <util/expr_reassociate.h>

void goto_reassociate(goto_functionst &goto_functions)
{
  Forall_goto_functions (f_it, goto_functions)
  {
    if (!f_it->second.body_available)
      continue;

    Forall_goto_program_instructions (i_it, f_it->second.body)
    {
      reassociate_arith(i_it->code);
      reassociate_arith(i_it->guard);
    }
  }

  goto_functions.update();
}
