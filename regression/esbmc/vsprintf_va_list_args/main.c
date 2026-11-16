#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

char b[16] = "abcdefghijklmno";

int fmt(const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  int r = vsprintf(b, f, ap);
  va_end(ap);
  return r;
}

int fmtn(size_t n, const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  int r = vsnprintf(b, n, f, ap);
  va_end(ap);
  return r;
}

int main(void)
{
  /* The arguments behind a va_list are only recovered where it is a pointer,
   * as in vsnprintf_valist_recovery_exact. */
  if (!__builtin_types_compatible_p(va_list, char *))
    return 0;

  assert(fmt("x%dy", 42) == 4);
  assert(b[0] == 'x' && b[1] == '4' && b[2] == '2' && b[3] == 'y');
  assert(b[4] == '\0' && b[5] == 'f');

  assert(fmtn(3, "%s-%d", "ab", 7) == 4);
  assert(b[0] == 'a' && b[1] == 'b' && b[2] == '\0' && b[3] == 'y');

  int x = nondet_int();
  __ESBMC_assume(x >= 0 && x < 10);
  int r = fmt("%x", x);
  assert(r == 1 && b[1] == '\0');
  return 0;
}
