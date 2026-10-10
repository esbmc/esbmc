#include <assert.h>
#include <stdarg.h>

int a, b, c, first, last;

void take(va_list ap)
{
  va_list cp;
  va_copy(cp, ap);
  a = va_arg(cp, int);
  b = va_arg(cp, int);
  va_end(cp);
  c = va_arg(ap, int);
}

int mid(int n, va_list outer, ...)
{
  va_list cp;
  va_copy(cp, outer);
  int r = va_arg(cp, int);
  va_end(cp);
  va_start(cp, outer);
  r = r * 10 + va_arg(cp, int);
  va_end(cp);
  return r;
}

void g(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  first = va_arg(ap, int);
  take(ap);
  last = va_arg(ap, int);
  va_end(ap);
}

int h(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int r = mid(0, ap, 9);
  va_end(ap);
  return r;
}

int main()
{
  g(4, 1, 2, 3, 4);
  assert(first == 1);
  assert(a == 2);
  assert(b == 3);
  assert(c == 2);
  assert(last == 3);
  assert(h(1, 7) == 79);
  return 0;
}
