/* A Kani binary: realloc cannot fail either, as under CBMC's
 * --no-malloc-may-fail. */
#include <stdlib.h>
int __KANI_marker;
int main(void)
{
  int *p = malloc(sizeof(int));
  int *q = realloc(p, 2 * sizeof(int));
  q[1] = 2;
  __CPROVER_assert(q[1] == 2, "the grown allocation holds the store");
  return 0;
}
