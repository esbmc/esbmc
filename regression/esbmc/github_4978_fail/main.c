#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Negative companion to GitHub #4978.  Bounding the vasprintf return (#5270)
// must not suppress overflow detection in the bb_verror_msg arithmetic shape:
// when another operand is genuinely unbounded the signed add can still overflow
// and ESBMC must report it.  Here applet_len is an unbounded nondet value, so
// `applet_len + used` overflows for real.  (used itself is bounded to [13,22]
// by the constant "%u" format, so the overflow is necessarily driven by
// applet_len -- which is exactly the point: the checker stays live.)
static int verror(const char *s, va_list p, int applet_len)
{
  char *msg;
  int used = vasprintf(&msg, s, p);
  if (used < 0)
    return 0;

  // Overflow-checked add with an unbounded applet_len: a real overflow exists.
  return applet_len + used + 3;
}

static void error_and_die(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  verror(s, p, __VERIFIER_nondet_int());
  va_end(p);
}

int main(void)
{
  error_and_die("unknown uid %u", (unsigned)__VERIFIER_nondet_int());
  return 0;
}
