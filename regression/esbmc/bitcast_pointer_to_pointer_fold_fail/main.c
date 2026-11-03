/* Read through the same bitcast, a pointer that may be NULL keeps its
 * claim. */
#include <stddef.h>

_Bool nondet_bool(void);
int x;

int main()
{
  int *p = nondet_bool() ? &x : NULL;
  void *q = *(void **)&p;
  __ESBMC_assert(q != NULL, "q may be NULL");
  return 0;
}
