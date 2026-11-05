/* The loop counts through **q. k-induction cannot havoc that cell, so its
 * inductive step starts where the loop started and does not reach the exit
 * within k iterations; that is no proof of non-termination. */
#include <stdlib.h>

int main(void)
{
  int x = 0;
  int *p = malloc(sizeof(int));
  if (!p)
    abort();
  int **q = &p;
  *p = 0;
  while (**q < 10)
  {
    x = 1 - x;
    **q = **q + 1;
  }
  free(p);
  return 0;
}
