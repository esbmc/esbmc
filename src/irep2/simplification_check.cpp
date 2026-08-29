#include <irep2/simplification_check.h>

namespace
{
simplification_check::checkert installed_checker;

/** Guards against a checker that itself simplifies -- create_solver and
 *  convert_ast both do -- recursing back into verify_rewrite forever.
 *  Thread-local: --parallel-solving simplifies on several threads, and one
 *  shared flag would both race and let every other thread's rewrites through
 *  unchecked while one thread held it. */
thread_local bool running = false;
} // namespace

void simplification_check::install(checkert checker)
{
  installed_checker = std::move(checker);
}

void simplification_check::clear()
{
  installed_checker = nullptr;
}

void simplification_check::detail::run(
  const expr2tc &before,
  const expr2tc &after)
{
  if (!installed_checker || running)
    return;

  running = true;
  // Restore on the way out however the checker leaves: a solver failure that
  // escapes as an exception must not wedge the hook off for the whole run.
  struct reset_on_exitt
  {
    ~reset_on_exitt()
    {
      running = false;
    }
  } reset;

  installed_checker(before, after);
}
