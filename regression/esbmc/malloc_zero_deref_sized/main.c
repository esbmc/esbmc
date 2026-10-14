/* Control for malloc_zero_deref_fail: the same access over an allocation of
 * the object's size, which is well defined and must verify. It is what
 * attributes that test's violation to the zero size rather than to the
 * cast or the dereference. */
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
  void *q = malloc(sizeof(struct pm));
  if ((unsigned long)q == 0UL)
    return 0;
  use((struct pm *)q);
  free(q);
  return 0;
}
