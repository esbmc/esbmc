/* ptr_sound_partial_update_fail: (soundness)
 * Struct with three fields; ensures requires all three to be set.
 * Body only sets the first two — third field stays at its initial value.
 * Must be VERIFICATION FAILED even though two out of three are correct.
 */
#include <stddef.h>

typedef struct { int a; int b; int c; } T;

void init(T *p)
{
  __ESBMC_requires(p != NULL);
  __ESBMC_ensures(p->a == 1 && p->b == 2 && p->c == 3);

  p->a = 1;
  p->b = 2;
  /* forgot: p->c = 3 */
}

int main()
{
  T t = {0, 0, 0};
  init(&t);
  return 0;
}
