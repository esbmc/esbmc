#include <goto-programs/goto_coverage.h>

// previously, we have skip the remove_sideefects in order
// to keep the original format of the guard.
// now we need to remove the sideeffects in the instrumented assertions
void goto_coverage_rm::remove_sideeffect()
{
  Forall_goto_functions (f_it, goto_functions)
  {
    if (f_it->second.body_available && f_it->first != "__ESBMC_main")
    {
      goto_programt &goto_program = f_it->second.body;

      // temporary storage for the result of remove_sideeffects
      goto_programt tmp_program;
      Forall_goto_program_instructions (it, goto_program)
      {
        if (
          it->is_assert() &&
          it->location.property().as_string() == "instrumented assertion" &&
          it->location.user_provided() == true)
        {
          expr2tc &temp_guard = it->guard;
          exprt guard = migrate_expr_back(temp_guard);
          remove_sideeffects(guard, tmp_program);

          // insert before it
          goto_program.destructive_insert(it, tmp_program);
          migrate_expr(guard, temp_guard);
        }
      }
    }
  }
  // re-calculation as we might have newly inserted instructions
  goto_functions.update();
}
