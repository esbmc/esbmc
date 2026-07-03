#include <stdarg.h>

extern int vasprintf(char **strp, const char *fmt, va_list ap);

/* Soundness guard for va_list recovery (design §4.4, GitHub #5012): `mid` is
 * itself variadic and va_copies a va_list received as a PARAMETER into a
 * fresh-looking local. The local's syntactic identity passes the own-local
 * gate, but it denotes `wrap`'s arguments, not `mid`'s: recovery must see the
 * copy (goto conversion lowers va_copy to a real assignment for pointer
 * va_lists) and decline via the freshness scan, leaving the return
 * unbounded, so the true length 6 ("x=aaaa") stays feasible and the
 * assertion FAILS. If the laundering went unseen, recovery would map mid's
 * own vararg ("zz" -> "x=zz", length 4), pin the return to 4, make
 * `used != 6` hold on every path, and this test would turn VERIFICATION
 * SUCCESSFUL -- catching the unsound mapping. */
static int mid(const char *fmt, va_list outer, ...)
{
  char *msg;
  va_list local;
  va_copy(local, outer);
  int used = vasprintf(&msg, fmt, local);
  va_end(local);
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
