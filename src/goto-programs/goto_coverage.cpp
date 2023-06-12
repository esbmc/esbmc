#include <goto-programs/goto_coverage.h>

int goto_coveraget::total_instrument = 0;

void goto_coveraget::make_asserts_false(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to false...");
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_assert())
        {
          const std::string old_comment = it->location.comment().as_string();
          it->guard = gen_false_expr();
          it->location.property("Instrumentation ASSERT(0)");
          if(old_comment != "")
            it->location.comment(
              "Instrumentation ASSERT(0) Converted, was " + old_comment);
          else
            it->location.comment("Instrumentation ASSERT(0) Converted");
          it->location.user_provided(true);
          total_instrument++;
        }
      }
    }
}

void goto_coveraget::add_false_asserts(goto_functionst &goto_functions)
{
  log_progress("Adding false assertions...");
  Forall_goto_functions(f_it, goto_functions)
    if(f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions(it, goto_program)
      {
        if(it->is_end_function())
        {
          insert_false_assert(goto_program, it);
          continue;
        }

        if((!is_true(it->guard) && it->is_goto()) || it->is_target())
        {
          it++; // add an assertion behind the instruciton
          insert_false_assert(goto_program, it);
          continue;
        }
      }

      goto_programt::targett it = goto_program.instructions.begin();
      insert_false_assert(goto_program, it);
    }
}

void goto_coveraget::insert_false_assert(
  goto_programt &goto_program,
  goto_programt::targett &it)
{
  goto_programt::targett t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = gen_false_expr();
  t->location = it->location;
  t->location.property("Instrumentation ASSERT(0)");
  t->location.comment("Instrumentation ASSERT(0) Added");
  t->location.user_provided(true);
  it = ++t;
  total_instrument++;
}

int goto_coveraget::get_total_instrument() const
{
  return total_instrument;
}
