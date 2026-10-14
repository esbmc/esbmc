/* Regression for #4442 review (no-data-race fallthrough boundary).
 *
 * Under --no-assertions WITHOUT --memory-leak-check the __assert_fail
 * lowering must remain a no-op so that user assert(cond) does NOT
 * silently become assume(cond).  If the path were truncated with
 * ASSUME false here, the NULL dereference on line 18 (the path that
 * SV-COMP no-data-race / overflow / pointer-check would otherwise
 * observe) would become unreachable and bugs hiding behind an
 * assertion would be silently dropped.
 *
 * SV-COMP no-data-race uses --no-pointer-check --no-bounds-check
 * --data-races-check-only --no-assertions: not the same flags as
 * here, but the regression mechanism is the same --- the post-assert
 * path must stay feasible for downstream checks.  We pin the
 * behaviour with the cheapest available downstream check
 * (pointer-deref, which is on by default and not suppressed by
 * --no-assertions).
 *
 * Expected: VERIFICATION FAILED (NULL pointer deref).
 */

extern void __assert_fail(const char *, const char *, unsigned int,
                          const char *)
    __attribute__((__nothrow__)) __attribute__((__noreturn__));

extern int nondet_int(void);

int main(void)
{
  int x = nondet_int();
  int *p = (int *)0;
  if (x == 42)
  {
    __assert_fail("0", __FILE__, __LINE__, __func__);
    return *p;
  }
  return 0;
}
