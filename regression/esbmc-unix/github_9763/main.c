// GitHub #9763: under --printf-check the va_list of a v* call is not a
// pointer argument to dereference.
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
  say("%s\n", "abc");
  return 0;
}
