#include <stdarg.h>
#include <assert.h>

double sum(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int total = 0;
  for (int i = 0; i < n; ++i)
    total += va_arg(ap, int);
  double scale = va_arg(ap, double);
  va_end(ap);
  return total * scale;
}

int main(void)
{
  assert(sum(3, 1, 2, 3, 2.0) == 12.0);
  return 0;
}
