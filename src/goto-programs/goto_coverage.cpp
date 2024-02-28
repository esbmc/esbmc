#include <goto-programs/goto_coverage.h>

int goto_coveraget::total_instrument = 0;
int goto_coveraget::total_assert_instance = 0;
std::unordered_set<std::string> total_cond_assert = {};

void goto_coveraget::make_asserts_false(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to false...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        const expr2tc old_guard = it->guard;
        if (it->is_assert())
        {
          it->guard = gen_false_expr();
          it->location.property("assertion");
          it->location.comment(from_expr(ns, "", old_guard));
          it->location.user_provided(true);
          total_instrument++;
        }
      }
    }
}

void goto_coveraget::make_asserts_true(goto_functionst &goto_functions)
{
  log_progress("Converting all assertions to false...");
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        const expr2tc old_guard = it->guard;
        if (it->is_assert())
        {
          it->guard = gen_true_expr();
          it->location.property("assertion");
          it->location.comment(from_expr(ns, "", old_guard));
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
          // insert an assert(0) as instrumentation BEFORE each instruction
          insert_assert(goto_program, it, gen_false_expr());
          continue;
        }

        if ((!is_true(it->guard) && it->is_goto()) || it->is_target())
        {
          it++; // add an assertion behind the instruciton
          insert_assert(goto_program, it, gen_false_expr());
          continue;
        }
      }

      goto_programt::targett it = goto_program.instructions.begin();
      insert_false_assert(goto_program, it);
    }
}

void goto_coveraget::insert_assert(
  goto_programt &goto_program,
  goto_programt::targett &it,
  const expr2tc &guard)
{
  goto_programt::targett t = goto_program.insert(it);
  t->type = ASSERT;
  t->guard = guard;
  t->location = it->location;
  t->location.property("assertion");
  t->location.comment(from_expr(ns, "", guard));
  t->location.user_provided(true);
  it = ++t;
  total_instrument++;
}

int goto_coveraget::get_total_instrument() const
{
  return total_instrument;
}

// Count the total assertion instances in goto level via goto-unwind api
// run the algorithm on the copy of the original goto program
void goto_coveraget::count_assert_instance(goto_functionst goto_functions)
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

std::unordered_set<std::string> goto_coveraget::get_total_cond_assert() const
{
  return total_cond_assert;
}

/*
  Condition Coverage: fault injection
  1. find condition statements, this includes the converted for_loop/while
  2. insert assertion instances before that statement.
  e.g.
    if (a >1)
  =>
    assert(!(a>1))
    assert(a>1)
    if(a>1)
  then run multi-property
*/
void goto_coveraget::add_cond_cov_assert(goto_functionst &goto_functions)
{
  Forall_goto_functions (f_it, goto_functions)
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;
      Forall_goto_program_instructions (it, goto_program)
      {
        // e.g. IF !(a > 1) THEN GOTO 3
        if (!is_true(it->guard) && it->is_goto())
        {
          const expr2tc old_guard = it->guard;
          insert_assert(goto_program, it, old_guard);
          std::string idf =
            from_expr(ns, "", old_guard) + "\t" + it->location.as_string();
          total_cond_assert.insert(idf);

          make_not(it->guard);
          insert_assert(goto_program, it, it->guard);
          idf = from_expr(ns, "", it->guard) + "\t" + it->location.as_string();
          total_cond_assert.insert(idf);
        }
      }
    }
}
