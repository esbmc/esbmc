/* Regression: PR #3777 — side-effect extraction in combined mode.
 *
 * The combined --loop-invariant mode uses raw invariant expressions in
 * ASSERT/ASSUME without re-inserting the DECL/FUNCTION_CALL that define
 * temporaries used in the invariant (unlike the legacy pass which uses
 * extract_and_remove_side_effects). So invariants that reference a function
 * call (e.g. in_range(x)) can see stale values after HAVOC.
 *
 * Same pattern as github_3711: invariant with in_range(qm) in the middle.
 * With correct side-effect extraction in combined mode: VERIFICATION SUCCESSFUL.
 * Without it we may get wrong FAILED or unsound SUCCESSFUL.
 */
#include <assert.h>

static int in_range(int n)
{
  return 0 <= n && n < 256;
}

int main(void)
{
  int qn = 0;
  int qm = 1;

  __ESBMC_loop_invariant(
    0 <= qn && qn < 256 && in_range(qm) && (qn != qm));
  for (int count = 0; count < 10; count++)
  {
    qn = (qn + 2) % 256;
    qm = (qm + 2) % 256;
  }

  assert(0 <= qn && qn < 256);
  return 0;
}
