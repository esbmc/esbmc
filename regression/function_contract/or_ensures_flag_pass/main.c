/* or_ensures_flag_pass:
 * ensures uses || to allow two valid postconditions depending on mode.
 * ensures(mode != 0 || p->x == 100):
 *   - when mode==0: left false, RIGHT side checked -> body must set p->x=100
 *   - when mode!=0: left true, short-circuit -> right NOT checked, always PASS
 *
 * Body correctly handles mode==0 by setting p->x=100.
 * ESBMC checks ALL nondet values of mode via --function harness.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, int mode)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(mode == 0 || mode == 1);
  __ESBMC_ensures(mode != 0 || p->x == 100);

  if (mode == 0)
    p->x = 100;
}

int main() { return 0; }
