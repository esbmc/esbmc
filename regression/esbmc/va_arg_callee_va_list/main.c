#include <assert.h>
#include <stdarg.h>

struct S
{
  int a, b;
};

static int take(va_list ap)
{
  struct S s = va_arg(ap, struct S);
  int k = va_arg(ap, int);
  return s.a + s.b + k;
}

static int mid(int n, va_list outer, ...)
{
  return va_arg(outer, int);
}

static int g(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int r = n ? take(ap) : mid(0, ap, 9);
  va_end(ap);
  return r;
}

int main()
{
  struct S x = {1, 2};
  assert(g(1, x, 4) == 7);
  assert(g(0, 3) == 3);
  return 0;
}
