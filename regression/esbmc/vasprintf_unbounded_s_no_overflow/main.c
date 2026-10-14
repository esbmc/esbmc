#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern char *__VERIFIER_nondet_charp(void);

// Regression for GitHub #5144: vasprintf with a %s conversion makes the
// output length statically unbounded.  Before the fix the return was modelled
// as nondet >= 0 (no upper bound), so `applet_len + used` appeared to overflow.
// The fix caps the return at INT_MAX/2 when the format is not soundly bounded,
// making the sum provably safe when applet_len is small.
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
  char *s = __VERIFIER_nondet_charp();
  int used = wrap("number %s is not in range", s);
  if (used < 0)
    return 0;
  int applet_len = 8;
  int sum = applet_len + used;
  return sum;
}
