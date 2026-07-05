#include <stdarg.h>
#include <string.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Regression for GitHub #4978 (SV-COMP busybox whoami-incomplete-1, no-overflow).
// Reduced from bb_verror_msg: the overflow-checked expression is
//   applet_len + used + strerr_len + msgeol_len + 3
// where every term is a signed int.  `used` is the vasprintf return value.
// Before #5270 wired vasprintf into symex_printf, its return was an
// unconstrained nondet, so the solver could pick used = INT_MAX and the signed
// add appeared to overflow -- a spurious counterexample (originally reported at
// k = 11 under k-induction).  With the wiring, the constant "%u" format is
// soundly bounded by printf_formatter to a small range ([13,22] here), so the
// sum is provably safe.  (The complementary INT_MAX/2 cap from #5278 covers the
// harder case of an unbounded format such as "%s"; see vasprintf_unbounded_s_*.)
//
// The 10-iteration applet_name loop drives k-induction to the k = 11 step where
// the original false alarm fired, which the simpler vasprintf_* tests do not.
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
  // Bounded applet name, matching the harness (10 nondet bytes + NUL).
  for (int i = 0; i < 10; ++i)
    applet_buf[i] = (char)__VERIFIER_nondet_int();
  applet_buf[10] = 0;

  error_and_die("unknown uid %u", (unsigned)__VERIFIER_nondet_int());
  return 0;
}
