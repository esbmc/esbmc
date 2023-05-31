#include <goto-programs/goto_coverage.h>

void make_asserts_false(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to false...");
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_assert())
        {
          it->guard = gen_false_expr();
          it->location.property("Instrumentation ASSERT(0)");
          it->location.comment("Instrumentation ASSERT(0)");
        }
      }
    }
}

void add_false_asserts(goto_functionst &goto_functions)
{
  log_progress("Adding false assertions...");
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available)
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_end_function())
        {
          insert_false_assert(goto_program, it);
        }

        if((it->is_goto() && !is_true(it->guard)) || it->is_target())
        {
          it++; // add an assertion behind the instruciton
          insert_false_assert(goto_program, it);
        }
      }

      goto_programt::targett it = goto_program.instructions.begin();
      insert_false_assert(goto_program, it);
    }
}

void insert_false_assert(
  goto_programt &goto_program,
  goto_programt::targett &it)
{
  goto_programt::targett t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = gen_false_expr();
  t->location = it->location;
  t->location.property("Instrumentation ASSERT(0)");
  t->location.comment("Instrumentation ASSERT(0)");
  it = ++t;
}
