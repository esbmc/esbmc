#include <stdarg.h>

static int through_copy(int n, ...)
{
  va_list ap, cp;
  va_start(ap, n);
  va_copy(cp, ap);
  int a = va_arg(cp, int);
  va_end(cp);
  va_end(ap);
  return a;
}

int main()
{
  __ESBMC_assert(through_copy(1, 7) == 7, "va_arg reads through the copy");
  return 0;
}
