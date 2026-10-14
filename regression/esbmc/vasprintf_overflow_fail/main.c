#include <stdarg.h>
#include <limits.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

// Negative companion to vasprintf_const_format_no_overflow: the modelled return
// length must NOT be under-approximated, or a real overflow on the return value
// would be missed (false negative). "unknown uid %u" bounds the return to
// [13, 22]; adding it to INT_MAX - 5 has a reachable signed overflow, so ESBMC
// must report VERIFICATION FAILED. Guards against re-introducing the unsound
// "modelled return is 0" behaviour for the va_list forms.
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
  int used = wrap("unknown uid %u", 7u);
  if (used < 0)
    return 0;
  int big = INT_MAX - 5;
  int sum = big + used; // overflows: used >= 13
  return sum;
}
