#include <stdarg.h>

/* va_copy from a va_list that was never started does not launder it —
 * consuming the copy is still undefined behaviour. */
int sum(int count, ...)
{
  va_list ap, aq;
  /* va_start(ap, count); -- deliberately omitted */
  va_copy(aq, ap);
  int total = 0;
  for (int i = 0; i < count; i++)
    total += va_arg(aq, int);
  return total;
}

int main()
{
  int r = sum(2, 3, 4);
  __ESBMC_assert(r == 7, "unreachable under correct varargs semantics");
  return 0;
}
