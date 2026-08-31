#include <stdarg.h>

static int first(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int a = va_arg(ap, int);
  va_end(ap);
  return a;
}

int main()
{
  __ESBMC_assert(first(1, 7) == 7, "va_arg reads the first argument");
  return 0;
}
