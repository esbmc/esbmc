/* Not a Kani binary: malloc may still fail. */
#include <stdlib.h>
int main(void)
{
  int *p = malloc(sizeof(int));
  *p = 1;
  __CPROVER_assert(*p == 1, "the allocation holds the store");
  return 0;
}
