#include <assert.h>

typedef __builtin_va_list va_list;
/* CIL prints this prototype (ldv-linux-4.2-rc1); clang refuses it. */
extern void __builtin_va_copy(__builtin_va_list, __builtin_va_list);

static int f(int n, ...)
{
  va_list ap, cp;
  __builtin_va_start(ap, n);
  int a = __builtin_va_arg(ap, int);
  __builtin_va_copy((va_list *)(&cp), ap);
  int b = __builtin_va_arg(cp, int);
  __builtin_va_end(cp);
  __builtin_va_end(ap);
  return a * 10 + b;
}

int main()
{
  /* The copy starts where ap stands, after the first argument. */
  assert(f(1, 7, 8) == 77);
  return 0;
}
