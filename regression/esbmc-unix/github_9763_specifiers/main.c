// GitHub #9763: the arguments of a v* call are behind its va_list, so the
// format is not checked against its operands.
#include <stdarg.h>
#include <stdio.h>

static void say(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vprintf(fmt, ap);
  va_end(ap);
}

int main(void)
{
  say("%s %d\n", "abc", 3);
  return 0;
}
