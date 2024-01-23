#include <goto-programs/goto_coverage.h>

int goto_coveraget::total_instrument = 0;
int goto_coveraget::total_assert_instance = 0;

void goto_coveraget::make_asserts_false(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to false...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        if (it->is_assert())
        {
          std::string cmt;
          const std::string loc = it->location.as_string();
          const std::string old_comment = it->location.comment().as_string();

          it->guard = gen_false_expr();
          it->location.property("Instrumentation ASSERT(0)");
          cmt = "Claim " + std::to_string(total_instrument + 1) + ": " +
                old_comment;
          it->location.comment(cmt);
          it->location.user_provided(true);
          total_instrument++;
        }
      }
    }
}

void goto_coveraget::add_false_asserts(goto_functionst &goto_functions)
{
  log_progress("Adding false assertions...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        if (it->is_end_function())
        {
          insert_false_assert(goto_program, it);
          continue;
        }

        if ((!is_true(it->guard) && it->is_goto()) || it->is_target())
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
  t->location.comment(
    "Claim " + std::to_string(total_instrument + 1) +
    ": Instrumentation ASSERT(0) Added");
  t->location.user_provided(true);
  it = ++t;
  total_instrument++;
}

int goto_coveraget::get_total_instrument() const
{
  return total_instrument;
}

// Obtain total assertion instances in goto level via goto-unwind api
// run the algorithm on the copy of the original goto program
void goto_coveraget::gen_assert_instance(goto_functionst goto_functions)
{
  // 1. execute goto uniwnd
  bounded_loop_unroller unwind_loops;
  unwind_loops.run(goto_functions);
  // 2. calculate the number of assertion instance
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        if (it->is_assert())
          total_assert_instance++;
      }
    }
}

int goto_coveraget::get_total_assert_instance() const
{
  return total_assert_instance;
}
