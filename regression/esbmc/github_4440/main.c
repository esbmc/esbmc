/* Regression for #4440.
 *
 * Reduced from c/coreutils-v9.5-units/seq_cmp_antisymmetry_cover_proof.i
 * (SV-COMP 26).  The cover_proof variant interposes a postcondition step
 * (fv_assert) between the local malloc and the unconditional cover_check
 * that ends in reach_error() -> __assert_fail() — the shape #4441's
 * regression (the cover_target variant) does not exercise.
 *
 * __assert_fail is __noreturn, so under --no-assertions the call must
 * still terminate the path; otherwise the end-of-main memcleanup walker
 * spuriously reports "forgotten memory" on heap that is only unreachable
 * on the post-noreturn fall-through path.  Fixed by #4442.
 *
 * The postcondition operand is derived from __VERIFIER_nondet_int so both
 * arms of fv_assert are reachable: result < 0 takes the inner reach_error
 * path, result >= 0 falls through to main's unconditional reach_error.
 * Both paths must be truncated by ASSUME(false) for verification to
 * succeed.
 *
 * Expected: VERIFICATION SUCCESSFUL on valid-memsafety semantics
 * (--memory-leak-check --no-reachable-memory-leak --no-assertions
 * --no-abnormal-memory-leak — the SV-COMP valid-memsafety wrapper flags).
 */

#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);
extern void __assert_fail(const char *, const char *, unsigned int,
                          const char *)
    __attribute__((__nothrow__)) __attribute__((__noreturn__));

void reach_error(void)
{
  __assert_fail("0", __FILE__, __LINE__, __func__);
}

static void fv_assert(int condition)
{
  if (!condition)
    reach_error();
}

static int result;

static void do_work(void)
{
  int *p = malloc(sizeof(int));
  if (!p)
    return;
  *p = __VERIFIER_nondet_int();
  result = *p;
}

static void postcond(void)
{
  fv_assert(result >= 0);
}

int main(void)
{
  do_work();
  postcond();
  reach_error();
  return 0;
}
