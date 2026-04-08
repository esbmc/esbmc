/* ptr_sound_wrong_field_fail: (soundness)
 * ensures(p->x == 42) but body sets p->y = 42, leaving p->x unchanged.
 * Classic "update wrong field" bug — must be VERIFICATION FAILED.
 */
#include <stddef.h>

typedef struct { int x; int y; } S;

void f(S *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p->x == 42);

  p->y = 42; /* wrong field */
}

int main()
{
  S s = {0, 0};
  f(&s);
  return 0;
}
