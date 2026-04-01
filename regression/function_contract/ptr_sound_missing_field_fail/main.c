/* ptr_sound_missing_field_fail: (soundness)
 * ensures requires BOTH p->x == 1 AND p->y == 2.
 * Body only sets p->x = 1, forgetting p->y.
 * Must be VERIFICATION FAILED.
 */
#include <stddef.h>

typedef struct { int x; int y; } S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p->x == 1 && p->y == 2);

  p->x = 1;
  /* forgot: p->y = 2 */
}

int main()
{
  S s = {0, 0};
  f(&s);
  return 0;
}
