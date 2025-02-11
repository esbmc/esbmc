#include <stdarg.h>
#include <stdio.h>

int sum(int count, ...)
{
  va_list args;
  va_start(args, count);

  int total = 0;
  for (int i = 0; i < count; i++)
    total += va_arg(args, int);

  va_end(args);

  return total;
}

int main()
{
  int result = sum(4, 1, 2, 3, 4);
  __ESBMC_assert(result == 10, "");

  return 0;
}
