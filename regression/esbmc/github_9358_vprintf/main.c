// #9358: a v* printf passes a va_list, not the %s argument, so --printf-check
// claims no NUL for it. --no-pointer-check: the pointer-argument dereference
// check also reads the va_list itself and reports it invalid.
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
  char tag[4] = {'a', 'b', 'c', 0};
  say("%s\n", tag);
  return 0;
}
