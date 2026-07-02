#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

/* va_list %s recovery (design §4.4, GitHub #5012): a v* call in the variadic
 * function's own frame, on its own unconsumed va_list, recovers the actual
 * arguments, so a string-literal %s pins the return to the exact length just
 * like a direct asprintf call. "hello %s!" with "world" formats to
 * "hello world!" (12 characters). Fails if recovery never fires (the return
 * would be unbounded and the equality unprovable). */
static int wrap(const char *fmt, ...)
{
  char *msg;
  va_list ap;
  va_start(ap, fmt);
  int used = vasprintf(&msg, fmt, ap);
  va_end(ap);
  return used;
}

int main(void)
{
  int used = wrap("hello %s!", "world");
  __ESBMC_assert(used == 12, "recovered literal %s pins the exact length");
  return 0;
}
