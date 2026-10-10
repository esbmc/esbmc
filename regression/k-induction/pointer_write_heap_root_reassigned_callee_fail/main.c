// As pointer_write_heap_root_callee_havoced_fail, through a pointer main's
// loop reassigns from g's result before writing through it.
#include <stdlib.h>
#include <assert.h>
extern int nondet_int(void);
extern void __VERIFIER_assume(int);
int *A;
int *g(void)
{
  int *x = A;
  int i = 0;
  while (i < 1)
  {
    x = x;
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
