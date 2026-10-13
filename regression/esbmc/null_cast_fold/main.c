/* `p != NULL` against a `T *` reaches the simplifier as `(T *)NULL`. The
 * non-null fold has to recognise it, or the claim goes to the solver. */
#include <stddef.h>

struct node
{
  unsigned long size;
};

int main()
{
  struct node *p = __ESBMC_alloca(sizeof(struct node));
  __ESBMC_assert(p != NULL, "an allocation is not NULL");
  return 0;
}
