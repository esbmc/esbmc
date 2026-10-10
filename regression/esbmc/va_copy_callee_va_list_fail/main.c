#include <assert.h>
#include <stdarg.h>

static int mid(int n, va_list outer, ...)
{
  va_list cp;
  va_copy(cp, outer);
  int r = va_arg(cp, int);
  va_end(cp);
  return r;
}

static int g(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int r = mid(0, ap, 9);
  va_end(ap);
  return r;
}

int main()
{
  assert(g(1, 3) == 9);
  return 0;
}
