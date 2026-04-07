/* or_ensures_two_branches_pass:
 * ensures has two complete branches via ||, each covering a different case.
 * ensures((select==0 && p->x==10 && p->y==20) || (select==1 && p->x==30 && p->y==40))
 * Body handles both cases correctly.
 * Tests || where BOTH sides reference multiple pointer fields simultaneously.
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
  else             { p->x = 30; p->y = 40; }
}

int main() { return 0; }
