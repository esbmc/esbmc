#include <stdarg.h>
#include <stdio.h>

/* va_list %s recovery (design §4.4, GitHub #5012) on the vsnprintf logging
 * idiom: the recovered literal %s pins the would-be length exactly, which is
 * independent of the size cap per C11 7.21.6.12. */
static int log_msg(char *buf, unsigned long n, const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int would_be = vsnprintf(buf, n, fmt, ap);
  va_end(ap);
  return would_be;
}

int main(void)
{
  char buf[64];
  int would_be = log_msg(buf, sizeof(buf), "hello %s!", "world");
  __ESBMC_assert(would_be == 12, "recovered literal %s pins would-be length");
  return 0;
}
