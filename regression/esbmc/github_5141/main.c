#include <stdarg.h>
#include <stdlib.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Regression test for GitHub #5141: after vasprintf the buffer pointer *strp
// must be a valid tracked heap allocation so that free(*strp) is sound.
// Before the fix, symex_printf did not model *strp, leaving it as a nondet
// value; ESBMC then reported a memsafety violation on free(msg) because the
// pointer was not in the tracked malloc set.
static void format_msg(const char *s, va_list p)
{
  char *msg;
  int used = vasprintf(&msg, s, p);
  if (used < 0)
    return;
  free(msg);
}

void wrapper(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  format_msg(fmt, ap);
  va_end(ap);
}

int main(void)
{
  wrapper("hello %d", __VERIFIER_nondet_int());
  return 0;
}
