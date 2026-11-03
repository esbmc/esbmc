#include <assert.h>
#include <stdarg.h>
#include <stdio.h>

static int put(char *b, const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  int r = vsprintf(b, f, ap);
  va_end(ap);
  return r;
}

static int nput(char *b, size_t n, const char *f, ...)
{
  va_list ap;
  va_start(ap, f);
  int r = vsnprintf(b, n, f, ap);
  va_end(ap);
  return r;
}

int main(void)
{
  char b[8] = "abcdefg";
  assert(put(b, "xy%%", 1) == 3);
  assert(b[0] == 'x' && b[1] == 'y' && b[2] == '%' && b[3] == '\0');
  assert(b[4] == 'e');

  char c[4] = "abc";
  assert(nput(c, 2, "pqr") == 3);
  assert(c[0] == 'p' && c[1] == '\0' && c[2] == 'c');
  assert(nput(NULL, 0, "pqr") == 3);
  return 0;
}
