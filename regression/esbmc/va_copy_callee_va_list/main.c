#include <assert.h>
#include <stdarg.h>

static int sum2(va_list ap)
{
  va_list cp;
  va_copy(cp, ap);
  int a = va_arg(cp, int);
  int b = va_arg(cp, int);
  va_end(cp);
  return a * 10 + b;
}

static int copy_then_read(va_list ap)
{
  va_list cp;
  va_copy(cp, ap);
  int a = va_arg(ap, int);
  int b = va_arg(cp, int);
  va_end(cp);
  return a * 10 + b;
}

static int f(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int r = n ? sum2(ap) : copy_then_read(ap);
  va_end(ap);
  return r;
}

int main()
{
  assert(f(1, 3, 4) == 34);
  assert(f(0, 5, 6) == 55);
  return 0;
}
