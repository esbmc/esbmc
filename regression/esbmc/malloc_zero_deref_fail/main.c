/* C17 7.22.3p1: when the requested size is zero the behaviour is
 * implementation-defined -- either a null pointer is returned, or the behaviour
 * is as if the size were some nonzero value, "except that the returned pointer
 * shall not be used to access an object". Reading an object through a
 * malloc(0) pointer is therefore undefined, and flagging it is correct.
 *
 * This shape is why #5397 reports false on
 * ldv-linux-3.14-races/linux-3.14--drivers--net--irda--nsc-ircc.ko.cil: the
 * harness calls ldv_xmalloc(0UL), assumes the result non-null, casts it to
 * struct pm_message * and dereferences it. The task's expected verdict is true,
 * but the access is undefined by the paragraph above -- so this test exists to
 * keep the check from being weakened to match that expectation.
 * See malloc_zero_deref_sized for the same access over a sized allocation. */
#include <stdlib.h>

struct pm
{
  int event;
};

static void use(struct pm *p)
{
  struct pm v = *p;
  (void)v;
}

int main(void)
{
  void *q = malloc(0);
  if ((unsigned long)q == 0UL)
    return 0;
  use((struct pm *)q);
  free(q);
  return 0;
}
