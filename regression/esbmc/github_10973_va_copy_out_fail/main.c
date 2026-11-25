// Copying p out with va_copy leaves p itself denoting f's own arguments.
#include <stdarg.h>
#include <stdio.h>

static void f(const char *s, ...)
{
  va_list p, q;
  va_start(p, s);
  va_copy(q, p);
  vprintf(s, p);
  va_end(q);
  va_end(p);
}

int main(void)
{
  char buf[1] = {'a'};
  f("%s", buf);
  return 0;
}
