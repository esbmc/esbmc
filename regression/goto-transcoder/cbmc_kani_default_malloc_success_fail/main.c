/* A Kani binary whose assertion fails although malloc cannot fail. */
#include <stdlib.h>
int __KANI_marker;
int main(void)
{
  int *p = malloc(sizeof(int));
  *p = 1;
  __CPROVER_assert(*p == 2, "the allocation holds 2");
  return 0;
}
