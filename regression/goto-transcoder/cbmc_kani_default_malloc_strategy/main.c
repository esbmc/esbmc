/* The assumption holds under an explicit strategy too. */
#include <stdlib.h>
int __KANI_marker;
int main(void)
{
  int *p = malloc(sizeof(int));
  *p = 1;
  __CPROVER_assert(*p == 1, "the allocation holds the store");
  return 0;
}
