/* #9892: a pointer stored into a flexible array member of a global. */
#include <stdlib.h>
int nondet_int(void);
struct E { int *p; };
struct S { int n; struct E arr[]; };
struct S g;
int *h;
int main()
{
  int *q = malloc(sizeof(int));
  int x = nondet_int();
  if (x > 0 && x < 0)
    g.arr[0].p = q;
  return 0;
}
