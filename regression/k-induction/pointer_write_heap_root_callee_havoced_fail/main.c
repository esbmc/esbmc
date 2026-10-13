// h writes through x after its own loop, whose inductive step havocs x, so
// main's loop cannot trace x to A through its assignment: the step would
// leave x unresolved, drop the write and prove *A != 7.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
extern void __VERIFIER_assume(int);
int *A;
int n;
void h(void)
{
  int *x = A;
  int i = 0;
  while (i < 1)
  {
    x = x;
    i++;
  }
  if (n >= 3)
    *x = 7;
}
int main()
{
  A = malloc(sizeof(int));
  *A = 0;
  while (nondet_int())
  {
    __VERIFIER_assume(*A != 7);
    h();
    assert(*A != 7);
    n++;
  }
  return 0;
}
