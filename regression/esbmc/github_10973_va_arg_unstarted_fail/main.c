#include <stdarg.h>

static void f(int n, ...)
{
  va_list p;
  (void)va_arg(p, int);
}

int main(void)
{
  f(1, 2);
  return 0;
}
