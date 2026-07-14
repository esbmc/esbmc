#include <stdarg.h>
#include <limits.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

/* Negative guard for va_list recovery (design §4.4, GitHub #5012): the
 * recovered return is the exact formatted length (12), so a sum engineered
 * to overflow at that value must still be reported. This stays FAILED under
 * the unbounded fallback too (any used > 5 overflows); what it pins is that
 * no future change confines the recovered return to values <= 5 -- the
 * under-approximation direction design §2 forbids. */
static int wrap(const char *fmt, ...)
{
  char *msg;
  va_list ap;
  va_start(ap, fmt);
  int used = vasprintf(&msg, fmt, ap);
  va_end(ap);
  return used;
}

int main(void)
{
  int used = wrap("hello %s!", "world"); /* exactly 12 */
  if (used < 0)
    return 0;
  int sum = (INT_MAX - 5) + used; /* overflows for any used > 5 */
  return sum != 0;
}
