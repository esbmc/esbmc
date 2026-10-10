// As pointer_write_heap_root_reassigned_callee_fail, with g's loop also
// entered by a conditional jump into its body.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
extern void __VERIFIER_assume(int);
int *A;
int F;
int *g(void)
{
  int *x = A;
  int i = 0;
  if (F)
    goto inside;
  while (i < 1)
  {
    x = x;
  inside:
    i++;
  }
  return x;
}
int main()
{
  A = malloc(sizeof(int));
  *A = 0;
  int n = 0;
  int *r;
  while (nondet_int())
  {
    r = g();
    __VERIFIER_assume(*A != 7);
    if (n >= 3)
      *r = 7;
    assert(*A != 7);
    n++;
  }
  return 0;
}
