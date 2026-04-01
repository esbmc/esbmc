/* or_ensures_two_branches_fail:
 * Same contract. select==1 branch has p->y=39 instead of 40.
 *
 * select==0: first disjunct (0==0 && x==10 && y==20) = TRUE -> pass
 * select==1: first disjunct (1==0 && ...) = FALSE
 *            second disjunct (1==1 && x==30 && 39==40) = FALSE
 *            FALSE || FALSE -> VERIFICATION FAILED
 *
 * Confirms || does NOT hide partial violations when the active branch
 * makes BOTH disjuncts false simultaneously.
 */
#include <stddef.h>

typedef struct { int x; int y; } S;

void f(S *p, int select)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_requires(select == 0 || select == 1);
  __ESBMC_ensures(
    (select == 0 && p->x == 10 && p->y == 20) ||
    (select == 1 && p->x == 30 && p->y == 40));

  if (select == 0) { p->x = 10; p->y = 20; }
  else             { p->x = 30; p->y = 39; } /* wrong: y should be 40 */
}

int main() { return 0; }
