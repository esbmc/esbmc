/* A pointer that may be NULL keeps its claim. */
#include <stddef.h>

_Bool nondet_bool(void);
int x;

int main()
{
  int *p = nondet_bool() ? &x : NULL;
  __ESBMC_assert(p != NULL, "p may be NULL");
  return 0;
}
