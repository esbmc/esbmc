#include <assert.h>
#include <stdarg.h>

static int second(int n, ...)
{
  va_list p;
  va_start(p, n);
  (void)va_arg(p, int);
  int y = va_arg(p, int);
  va_end(p);
  return y;
}

int main(void)
{
  assert(second(2, 10, 20) == 10);
  return 0;
}
