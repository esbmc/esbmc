#include <stdarg.h>
#include <string.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Regression for GitHub #4979 (SV-COMP busybox whoami-incomplete-2, no-overflow).
// Identical bb_verror_msg overflow shape as #4978: the whoami applet's call is
//   bb_error_msg_and_die("unknown uid %u", (unsigned)uid)
// so `used = vasprintf(&msg, "unknown uid %u", p)` is bounded by
// printf_formatter to [13,22] once #5270 wired vasprintf into symex_printf.
// Before that the return was an unconstrained nondet and the signed add
// `applet_len + used + strerr_len + msgeol_len + 3` appeared to overflow -- the
// spurious counterexample originally reported at k = 11 under k-induction.
//
// This pins the whoami-incomplete-2 variant specifically (the -1 variant is
// #4978); both exercise the constant "%u" return-length bound.
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

  error_and_die("unknown uid %u", (unsigned)__VERIFIER_nondet_int());
  return 0;
}
