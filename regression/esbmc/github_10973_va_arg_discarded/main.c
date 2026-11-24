// A discarded va_arg still consumes its argument (C11 7.16.1.1p2).
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

static int second(int n, ...)
{
  va_list p;
  va_start(p, n);
  (void)va_arg(p, int);
  int y = va_arg(p, int);
  va_end(p);
  return y;
}

static void skip_one(const char *s, ...)
{
  va_list p;
  va_start(p, s);
  va_arg(p, int);
  vprintf(s, p);
  va_end(p);
}

int main(void)
{
  assert(second(2, 10, 20) == 20);
  skip_one("%s", 1, "ok");
  return 0;
}
