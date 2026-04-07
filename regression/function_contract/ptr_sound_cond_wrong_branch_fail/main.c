/* ptr_sound_cond_wrong_branch_fail: (soundness)
 * Function should write a>0 ? 1 : -1 (sign function).
 * Body always writes 1, which is wrong when a<=0.
 * ensures correctly specifies sign semantics -> VERIFICATION FAILED.
 * Tests that a conditional ensures clause catches wrong branch logic.
 */
#include <stddef.h>

void sign(int a, int *result)
{
  __ESBMC_requires(result != NULL);
  __ESBMC_ensures(*result == (a > 0 ? 1 : -1));

  *result = 1; /* wrong: ignores a <= 0 case */
}

int main()
{
  int r;
  sign(-5, &r);
  return 0;
}
