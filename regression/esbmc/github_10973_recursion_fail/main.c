// The nested call's arguments are not the outer activation's.
#include <stdarg.h>
#include <stdio.h>

static void wrap(int depth, const char *s, ...)
{
  if (depth)
    wrap(0, "%s\n", "inner");
  va_list p;
  va_start(p, s);
  vprintf(s, p);
  va_end(p);
}

int main(void)
{
  wrap(1, "%s%s\n", "ab");
  return 0;
}
