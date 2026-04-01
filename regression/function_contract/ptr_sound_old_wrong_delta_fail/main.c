/* ptr_sound_old_wrong_delta_fail: (soundness)
 * ensures claims delta +5 from old(*p), but body adds only 3.
 * Must be VERIFICATION FAILED for ALL nondet initial values of *p.
 * Tests that old-based ensures cannot be trivially satisfied.
 */
#include <stddef.h>

void f(int *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(*p == __ESBMC_old(*p) + 5); /* claims +5 */

  *p = *p + 3; /* actually +3 */
}

int main()
{
  int x = 10;
  f(&x);
  return 0;
}
