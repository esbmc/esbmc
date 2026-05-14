/* Regression for #4442 follow-up.
 *
 * Reduced from SV-COMP 26 list-ext-properties/simple-ext.c, which
 * regressed from correct-false to wrong-true after #4441's noreturn
 * fix: __assert_fail's ASSUME false truncates the path before
 * end-of-main, so the heap leak that valid-memcleanup expects to
 * report is never checked.
 *
 * Expected: VERIFICATION FAILED (forgotten memory) on valid-memcleanup
 * semantics (--memory-leak-check --no-assertions).  The malloc'd
 * buffer is never freed and main never returns normally — abort()
 * already invokes __ESBMC_memory_leak_checks() on its path; we
 * mirror that for __assert_fail.
 */

#include <stdlib.h>

extern void __assert_fail(const char *, const char *, unsigned int,
                          const char *)
    __attribute__((__nothrow__)) __attribute__((__noreturn__));

void reach_error(void)
{
  __assert_fail("0", __FILE__, __LINE__, __func__);
}

int main(void)
{
  char *p = malloc(8);
  if (!p)
    return 0;
  reach_error();
  return 0;
}
