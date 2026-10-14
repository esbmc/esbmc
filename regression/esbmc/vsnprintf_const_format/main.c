#include <stdarg.h>

extern int
vsnprintf(char *str, unsigned long size, const char *fmt, va_list ap);
extern int __VERIFIER_nondet_int(void);

// Exercises the format-argument-index-2 dispatch path (vsnprintf). The constant
// format "x=%d" bounds the return (the would-be length) to [3, 13]: 2 literal
// chars plus 1..11 for %d. The signed sum below is therefore bounded and must
// not overflow.
static int wrap(char *buf, unsigned long n, const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int r = vsnprintf(buf, n, fmt, ap);
  va_end(ap);
  return r;
}

int main(void)
{
  char buf[64];
  int r = wrap(buf, sizeof(buf), "x=%d", __VERIFIER_nondet_int());
  if (r < 0)
    return 0;
  int base = 100;
  int sum = base + r;
  return sum;
}
