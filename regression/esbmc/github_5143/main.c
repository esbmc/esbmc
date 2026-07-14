#include <stdarg.h>
#include <string.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Regression for GitHub #5143 (SV-COMP busybox usleep-1, no-overflow).
// Same bb_verror_msg overflow shape as #4978, but the usleep applet's failing
// call uses a *non-literal "%s"* format:
//   bb_error_msg_and_die("invalid number '%s'", numstr)
// Unlike the constant "%u" case (#4978/#4979) the "%s" argument has no static
// length, so printf_formatter cannot give a tight bound.  This is the harder
// path: #5278 caps the modelled return at INT_MAX/2 so it stays a sound
// over-approximation (R >= true length) while keeping the signed add
// `applet_len + used + strerr_len + msgeol_len + 3` provably non-overflowing
// for the small operands here.  Before #5270/#5278 `used` was an unconstrained
// nondet and the add reported a spurious overflow at k = 11 under k-induction.
static const char *msg_eol = "\n";
static char applet_buf[11];
static char numbuf[8];

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

static void error_and_die(const char *s, ...)
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

  // Bounded, NUL-terminated numeric string passed through a non-literal "%s".
  for (int i = 0; i < 7; ++i)
    numbuf[i] = (char)('0' + (__VERIFIER_nondet_int() & 7));
  numbuf[7] = 0;

  error_and_die("invalid number '%s'", numbuf);
  return 0;
}
