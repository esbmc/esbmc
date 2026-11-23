/* An ESBMC goto-binary is never a Kani binary, whatever it declares, so malloc
 * may still fail. */
#include <stdlib.h>
int __KANI_marker;
int main(void)
{
  int *p = malloc(sizeof(int));
  *p = 1;
  __ESBMC_assert(*p == 1, "the allocation holds the store");
  return 0;
}
