#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

static void f(const char *s, ...)
{
  va_list p;
  char *msg;
  va_start(p, s);
  if (vasprintf(&msg, s, p) >= 0)
    free(msg);
  va_end(p);
}

int main(void)
{
  char buf[2] = {'a', 0};
  f("%s %d %.1s %p\n", buf, 3, "xy", (void *)buf);
  return 0;
}
