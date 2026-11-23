/* "__KANI_" only as a string: the strategy default applies, but no symbol
 * names Kani, so malloc may still fail. */
#include <stdlib.h>
const char *name = "__KANI_";
int main(void)
{
  int *p = malloc(sizeof(int));
  *p = 1;
  __CPROVER_assert(*p == 1, "the allocation holds the store");
  return 0;
}
