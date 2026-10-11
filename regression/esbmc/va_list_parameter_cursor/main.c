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

static int g(int n, ...)
{
  va_list ap, aq;
  va_start(ap, n);
  va_copy(aq, ap);
  int a = va_arg(aq, int);
  va_end(aq);
  int r = copy_then_parameter(ap);
  va_end(ap);
  return 100 * a + r;
}

int main()
{
  assert(f(2, 1, 2) == 11);
  assert(g(2, 3, 4) == 333);
  return 0;
}
