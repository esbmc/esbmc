#include <stdarg.h>
#include <string.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Regression for GitHub #4977 (SV-COMP busybox sync-2, no-overflow).
// Same bb_verror_msg overflow shape as #4978, but the sync applet's call is
//   bb_error_msg("ignoring all arguments")
// i.e. a *constant format with no conversion specifier*.  vasprintf then
// returns exactly strlen("ignoring all arguments") == 22, the simplest case of
// the printf_formatter return-length bound wired up by #5270.  Before that
// wiring `used` was an unconstrained nondet, so the solver could pick
// used = INT_MAX and the signed add `applet_len + used + ...` appeared to
// overflow -- the spurious counterexample originally reported at k = 11 under
// k-induction.  With the constant-length bound the sum is provably safe.
//
// The 10-iteration applet_name loop drives k-induction to the k = 11 step where
// the original false alarm fired.
static const char *msg_eol = "\n";
static char applet_buf[11];

static int verror(const char *s, va_list p, const char *strerr)
{
  char *msg;
  int used = vasprintf(&msg, s, p);
  if (used < 0)
    return 0;

  int applet_len = (int)(strlen(applet_buf) + 2);
  int strerr_len = (int)(strerr ? strlen(strerr) : 0);
  int msgeol_len = (int)strlen(msg_eol);

  // Overflow-checked add: must not report a spurious signed overflow.
  return applet_len + used + strerr_len + msgeol_len + 3;
}

static void error_msg(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  verror(s, p, (const char *)0);
  va_end(p);
}

int main(void)
{
  for (int i = 0; i < 10; ++i)
    applet_buf[i] = (char)__VERIFIER_nondet_int();
  applet_buf[10] = 0;

  // Constant format, no varargs (the sync applet's "ignoring all arguments").
  error_msg("ignoring all arguments");
  return 0;
}
