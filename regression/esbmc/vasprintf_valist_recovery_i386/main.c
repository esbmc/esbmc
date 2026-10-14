#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

/* Companion to vasprintf_valist_recovery_exact that pins the pointer-va_list
 * recovery path on a target where va_list IS a plain pointer (i386, forced by
 * --32). On the default x86_64 target va_list is a struct array, so the sister
 * test's assertion is gated out and the pointer path never runs on Linux CI --
 * a regression there (e.g. a stray &ap reference disabling the freshness scan)
 * only surfaced on the Windows runner. Running the same scenario under --32
 * exercises that path here so the class of bug is caught on every target. */
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
  /* "hello world!" is 12 characters; recovery must pin the exact length. */
  __ESBMC_assert(used == 12, "pointer-va_list recovery pins the exact length");
  return 0;
}
