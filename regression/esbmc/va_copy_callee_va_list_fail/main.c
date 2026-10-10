#include <assert.h>
#include <stdarg.h>

int take(va_list ap)
{
  va_list cp;
  va_copy(cp, ap);
  int r = va_arg(cp, int);
  va_end(cp);
  return r;
}

int g(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int r = take(ap);
  va_end(ap);
  return r;
}

int main()
{
  assert(g(1, 3) == 0);
  return 0;
}
