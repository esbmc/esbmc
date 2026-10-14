#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

/* Soundness guard for va_list recovery (design §4.4, GitHub #5012): a va_arg
 * was consumed before the v* call, so the frame cursor no longer equals the
 * activation base. A va_start rewind is invisible to symex, making the
 * specifier-to-argument alignment unknowable: recovery must decline and the
 * return stays unbounded, keeping the true length 3 ("abc") feasible, so the
 * assertion FAILS. If recovery wrongly fired from the cursor, the return
 * would pin to 3, `used != 3` would be unsatisfiable-to-violate, and this
 * test would turn VERIFICATION SUCCESSFUL -- catching the over-eager gate. */
static int wrap(const char *fmt, ...)
{
  char *msg;
  va_list ap;
  va_start(ap, fmt);
  int first = va_arg(ap, int);
  (void)first;
  int used = vasprintf(&msg, "%s", ap);
  va_end(ap);
  return used;
}

int main(void)
{
  int used = wrap("unused", 7, "abc"); /* %s consumes "abc": true length 3 */
  __ESBMC_assert(used != 3, "unbounded return must keep 3 feasible");
  return 0;
}
