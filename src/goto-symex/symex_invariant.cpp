#include <cstdio>
#include <cstdlib>
#include <goto-symex/symex_invariant.h>
#include <util/message/message.h>

void symex_invariant_violated(
  const char *file,
  unsigned line,
  const char *function,
  const char *condition,
  const char *reason)
{
  log_error(
    "goto-symex invariant violated in {} ({}:{})\n"
    "  condition: {}\n"
    "  {}\n"
    "This is a defect in ESBMC, not in the program under analysis. Please "
    "report it at https://github.com/esbmc/esbmc/issues",
    function,
    file,
    line,
    condition,
    reason);
  // abort() is not required to flush open streams.
  fflush(nullptr);
  abort();
}
