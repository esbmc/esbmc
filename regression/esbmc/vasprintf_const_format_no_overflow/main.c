#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern unsigned int __VERIFIER_nondet_uint(void);

// Mirrors busybox bb_verror_msg (GitHub #4979): a variadic wrapper forwards a
// compile-time-constant format through a va_list to vasprintf, then uses the
// return value in a signed-int size computation. With vasprintf wired into the
// printf-family return-length model, "unknown uid %u" bounds the return to
// [13, 22] (12 literal chars + 1..10 for %u), so the sum cannot overflow.
// Before the fix, vasprintf had no operational model and the return was an
// unconstrained nondet int (~INT_MAX), producing a spurious arithmetic-overflow
// alarm. This is the Phase-2 functional-contract regression: reverting the
// wiring makes `used` unconstrained again and this test FAILS.
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
  int used = wrap("unknown uid %u", __VERIFIER_nondet_uint());
  if (used < 0)
    return 0;
  int applet_len = 8; // bounded sibling terms, as in the benchmark
  int msgeol_len = 1;
  int sum = applet_len + used + msgeol_len + 3;
  return sum;
}
