#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Regression test for GitHub #5139: with --add-symex-value-sets, an unmodeled
// *strp from vasprintf caused the resulting nondet pointer to appear in the
// value-sets of other pointers, producing spurious memsafety false alarms on
// subsequent pointer operations. The fix models *strp as a fresh tracked heap
// allocation, eliminating the aliasing.
static void format_and_use(const char *s, va_list p)
{
  char *msg;
  int used = vasprintf(&msg, s, p);
  if (used < 0)
    return;

  // Appending some extra length — mirrors the busybox bb_verror_msg pattern
  // that triggered the original false alarm.
  int extra = __VERIFIER_nondet_int();
  if (extra < 0 || extra > 100)
    extra = 0;
  char *buf = (char *)realloc(msg, (size_t)(used + extra + 1));
  if (buf)
    free(buf);
  else
    free(msg);
}

void wrapper(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  format_and_use(fmt, ap);
  va_end(ap);
}

int main(void)
{
  wrapper("value: %d", __VERIFIER_nondet_int());
  return 0;
}
