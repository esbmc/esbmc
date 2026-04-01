/* ptr_sound_two_ptrs_miss_one_fail: (soundness)
 * Two pointer params; ensures requires both p->x==10 and q->x==20.
 * Body only sets p->x=10 and forgets q->x.
 * Must be VERIFICATION FAILED because q->x stays at its initial nondet value.
 */
#include <stddef.h>

typedef struct { int x; } S;

void f(S *p, S *q)
{
  __ESBMC_requires(p != NULL && q != NULL);
  __ESBMC_ensures(p->x == 10 && q->x == 20);

  p->x = 10;
  /* forgot: q->x = 20 */
}

int main()
{
  S a = {0}, b = {0};
  f(&a, &b);
  return 0;
}
