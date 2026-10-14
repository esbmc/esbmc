#include <assert.h>
#include <stdarg.h>

static int copy_then_parameter(va_list ap)
{
  va_list cp;
  va_copy(cp, ap);
  int a = va_arg(cp, int);
  va_end(cp);
  int b = va_arg(ap, int);
  return 10 * a + b;
}

static int f(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int r = copy_then_parameter(ap);
  va_end(ap);
  return r;
}

int main()
{
  assert(f(2, 1, 2) == 12);
  return 0;
}
