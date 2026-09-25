#include <goto-programs/remove_library_assertions.h>
#include <langapi/mode.h>
#include <util/config/config.h>
#include <util/message/message.h>

void remove_library_assertions(goto_functionst &goto_functions)
{
  // The Python frontend also hides every function imported from one of the
  // user's own modules (converter_funcdef.cpp), so there a hidden body does
  // not imply "operational model" and dropping its assertions would discard
  // properties the user wrote.
  if (config.language.lid == language_idt::PYTHON)
  {
    log_warning(
      "--no-library-assertions is unsupported for Python: ESBMC cannot tell "
      "its own models apart from imported user modules");
    return;
  }

  unsigned count = 0;

  for (auto &fun_it : goto_functions.function_map)
    if (fun_it.second.body.hide)
      for (auto &instruction : fun_it.second.body.instructions)
        if (instruction.is_assert())
        {
          instruction.make_skip();
          count++;
        }

  log_status("Removed {} library assertion(s)", count);
}
