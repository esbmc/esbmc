/* CBMC lowers va_start into a side_effect that walks the stack for varargs;
   ESBMC models varargs as discrete per-call symbols, so it cannot yet support
   this on the --binary path. It must decline with a clean error, never abort()
   (roadmap §4.4). */
#include <stdarg.h>
int sum(int n, ...)
{
  va_list ap;
  va_start(ap, n);
  int s = 0;
  for (int i = 0; i < n; i++)
    s += va_arg(ap, int);
  va_end(ap);
  return s;
}
int main(void)
{
  __CPROVER_assert(sum(3, 10, 20, 30) == 60, "variadic sum");
  return 0;
}
