#include <stdarg.h>
#include <stdio.h>

static void put(char *b, const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  vsprintf(b, f, ap);
  va_end(ap);
}

int main(void)
{
  char b[4];
  put(b, "hello");
  return 0;
}
