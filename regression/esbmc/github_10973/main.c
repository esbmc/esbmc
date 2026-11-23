// GitHub #10973: --printf-check reads a %s argument passed through a va_list.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static void f(const char *s, ...)
{
  va_list p;
  char *msg;
  va_start(p, s);
  vasprintf(&msg, s, p);
  va_end(p);
}

int main(void)
{
  char buf[1] = {'a'};
  f("%s", buf);
  return 0;
}
