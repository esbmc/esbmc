#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

/* Soundness guard for va_list recovery (design §4.4, GitHub #5012): `mid` is
 * itself variadic but forwards a va_list received as a PARAMETER, i.e. the
 * arguments it denotes belong to `wrap`'s activation, not to `mid`'s.
 * Recovery must decline (parameter check), leaving the return unbounded, so
 * the true length 6 ("x=aaaa") stays feasible and the assertion FAILS.
 * If recovery wrongly mapped mid's own vararg ("zz" -> "x=zz", length 4),
 * the return would pin to 4, `used != 6` would hold on every path, and this
 * test would turn VERIFICATION SUCCESSFUL -- catching the unsound mapping. */
static int mid(const char *fmt, va_list outer, ...)
{
  char *msg;
  int used = vasprintf(&msg, fmt, outer);
  return used;
}

static int wrap(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int used = mid(fmt, ap, "zz");
  va_end(ap);
  return used;
}

int main(void)
{
  int used = wrap("x=%s", "aaaa"); /* true length 6 */
  __ESBMC_assert(used != 6, "unbounded return must keep 6 feasible");
  return 0;
}
