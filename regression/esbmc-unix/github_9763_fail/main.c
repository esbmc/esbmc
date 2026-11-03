// GitHub #9763: a direct printf next to a v* call is still checked.
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
  printf("%s %d\n", "abc");
  return 0;
}
