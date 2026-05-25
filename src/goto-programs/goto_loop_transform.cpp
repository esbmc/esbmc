#include <goto-programs/goto_loop_transform.h>

void goto_loop_transformt::run()
{
  Forall_goto_functions (it, goto_functions)
  {
    if (!visit_function(it->first, it->second))
      continue;

    goto_loopst loops(it->first, goto_functions, it->second);
    for (auto &loop : loops.get_loops())
    {
      if (!should_transform_loop(loop))
        continue;
      transform_loop(it->first, it->second, loop);
    }

    after_function(it->first, it->second);
  }
  goto_functions.update();
  finalize();
}

bool goto_loop_transformt::visit_function(
  const irep_idt & /*function_name*/,
  const goto_functiont &function) const
{
  return function.body_available;
}

bool goto_loop_transformt::should_transform_loop(const loopst &loop) const
{
  return !loop.get_modified_loop_vars().empty();
}
