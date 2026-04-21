/* or_ensures_nested_fail:
 * Same nested || contract, but flag==2 branch adds 3 instead of 5.
 * flag==0,1: short-circuit, vacuously pass.
 * flag==2:   both left sides FALSE, checks *p == old(*p)+5.
 *            body gives old+3 != old+5 -> VERIFICATION FAILED.
 *
 * Confirms old() at depth-2 in || is actually checked and can catch bugs.
 */
#include <stddef.h>

void f(int *p, int flag)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(flag == 0 || flag == 1 || flag == 2);
  __ESBMC_ensures(flag == 0 || flag == 1 || *p == __ESBMC_old(*p) + 5);

  if (flag == 2)
    *p = *p + 3; /* wrong: should be +5 */
}

int main() { return 0; }
