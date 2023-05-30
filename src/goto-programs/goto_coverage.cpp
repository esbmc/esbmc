#include <goto-programs/goto_coverage.h>

void make_assert_false(goto_functionst &goto_functions)
{
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_assert())
        {
          it->guard = gen_false_expr();
        }
      }
    }
}

void add_false_assert(goto_functionst &goto_functions)
{
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_return() || it->is_end_function())
        {
          goto_programt::targett t = goto_program.insert(it);
          t->type = ASSERT;
          t->guard = gen_false_expr();
          t->location = it->location;
          it = ++t;
        }
      }
    }
}
