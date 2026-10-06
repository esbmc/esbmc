#include <assert.h>
#include <stdarg.h>

int copy_first(int n, ...)
{
  va_list ap, aq;
  va_start(ap, n);
  va_copy(aq, ap);
  int a = va_arg(ap, int);
  int b = va_arg(aq, int);
  va_end(aq);
  va_end(ap);
  return a * 10 + b;
}

int restart(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int a = va_arg(ap, int);
  va_end(ap);
  va_start(ap, n);
  int b = va_arg(ap, int);
  va_end(ap);
  return a * 10 + b;
}

int two_lists(int n, ...)
{
  va_list ap, bp;
  va_start(ap, n);
  va_start(bp, n);
  int a = va_arg(ap, int);
  int b = va_arg(bp, int);
  va_end(bp);
  va_end(ap);
  return a * 10 + b;
}

int main()
{
  assert(copy_first(1, 3, 4) == 33);
  assert(restart(1, 3, 4) == 33);
  assert(two_lists(1, 5, 6) == 55);
}
