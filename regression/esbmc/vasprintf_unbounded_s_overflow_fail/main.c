#include <stdarg.h>
#include <limits.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern char *__VERIFIER_nondet_charp(void);

// Negative companion to vasprintf_unbounded_s_no_overflow: the modelled return
// for an unbounded %s format must NOT be under-approximated, or a real
// arithmetic overflow on the return value would be missed.  The cap is
// INT_MAX/2, so adding any base > INT_MAX/2 must still produce a reachable
// overflow.  Guards against re-introducing the "model return as 0" behaviour.
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
  int used = wrap("value %s", s);
  if (used < 0)
    return 0;
  int base = INT_MAX - 100; // base + INT_MAX/2 > INT_MAX, so overflow is reachable
  int sum = base + used;
  return sum;
}
