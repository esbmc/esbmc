/* Regression for #4441.
 *
 * Reduced from c/coreutils-v9.5-units/seq_cmp_antisymmetry_cover_target.i
 * (SV-COMP 26).  __assert_fail is __noreturn, so under --no-assertions the
 * call must still terminate the path; otherwise the end-of-main memcleanup
 * walker spuriously reports "forgotten memory" on heap that is only
 * unreachable on the post-noreturn fall-through path.
 *
 * Expected: VERIFICATION SUCCESSFUL on valid-memsafety semantics
 * (--memory-leak-check --no-reachable-memory-leak --no-assertions
 * --no-abnormal-memory-leak — same flags SV-COMP's valid-memsafety
 * wrapper uses; --no-abnormal-memory-leak suppresses leak checks at
 * abnormal-termination points like abort() / __assert_fail).
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
