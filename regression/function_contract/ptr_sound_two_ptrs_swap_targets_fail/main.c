/* ptr_sound_two_ptrs_swap_targets_fail: (soundness)
 * Body sets p->x=20 and q->x=10 (targets swapped).
 * ensures requires p->x==10 and q->x==20 — opposite of what body does.
 * Must be VERIFICATION FAILED.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(p != NULL && q != NULL);
  __ESBMC_ensures(p->x == 10 && q->x == 20);

  p->x = 20; /* swapped */
  q->x = 10; /* swapped */
}

int main()
{
  S a = {0}, b = {0};
  f(&a, &b);
  return 0;
}
