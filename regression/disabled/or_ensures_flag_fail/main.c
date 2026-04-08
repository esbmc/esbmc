/* or_ensures_flag_fail:
 * Same contract as or_ensures_flag_pass, but body sets p->x=99 when mode==0.
 * When mode==0: ensures = false || (99 == 100) = FALSE -> VERIFICATION FAILED.
 * Confirms short-circuit does NOT hide the violation when the active branch
 * (mode==0) makes the left side FALSE and the right side wrong.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, int mode)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(mode == 0 || mode == 1);
  __ESBMC_ensures(mode != 0 || p->x == 100);

  if (mode == 0)
    p->x = 99; /* wrong: should be 100 */
}

int main() { return 0; }
