// realloc copies the object the pointer points to, whichever of several it
// may be (C17 7.22.3.5p2).
#include <assert.h>
#include <stdlib.h>
int nondet_int();
int main()
{
  int *a = malloc(2 * sizeof(int));
  int *b = malloc(2 * sizeof(int));
  if (!a || !b)
    return 0;
  a[0] = 1;
  b[0] = 2;
  int pick_b = nondet_int();
  int *p = pick_b ? b : a;
  int *q = realloc(p, 4 * sizeof(int));
  if (!q)
    return 0;
  assert(pick_b || q[0] != 1);
  free(q);
  free(pick_b ? a : b);
  return 0;
}
