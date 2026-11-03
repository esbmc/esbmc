// m holds the address of an object main allocated before f's loop, so the
// write through it is no object the iteration allocated. Unless the inductive
// step havocs that object, *p keeps its pre-loop 0 and the step proves
// *(int *)m < 3.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
void f(long m)
{
  int *r;
  while (nondet_int())
  {
    r = (int *)m;
    *r = *r + 1;
    assert(*(int *)m < 3);
  }
}
int main()
{
  int *p = malloc(sizeof(int));
  *p = 0;
  f((long)p);
  return 0;
}
