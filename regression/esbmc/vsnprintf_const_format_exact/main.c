#include <stdio.h>
#include <stdarg.h>

// Regression test: vsnprintf was not recognised as printf-family before G-D
// wiring. ESBMC now routes it through symex_printf; a constant format with no
// conversion specifiers returns an exact length.  The addition cannot overflow,
// so verification must succeed.
static int wrap(char *buf, size_t cap, const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int r = vsnprintf(buf, cap, fmt, ap);
  va_end(ap);
  return r;
}

int main(void)
{
  char buf[64];
  int n = wrap(buf, sizeof(buf), "hello"); // exact length 5
  int base = 10;
  int t = base + n; // 15, no overflow
  return t;
}
