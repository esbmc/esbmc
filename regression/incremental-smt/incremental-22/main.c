#include <assert.h>
int nondet_int();
int main()
{
  int a = nondet_int();
  int b = 0;
  int *c = &a;
  int *d = &b;
  unsigned n = 0;
  // The loop writes b only through d, in iteration 10. Unhavoced, b would keep
  // its 0 in an inductive step that starts at iteration 20 and the step would
  // prove the assertion; with b havoced the base case finds the violation.
  while(1)
  {
    *c = *c + 1;
    if(n == 10)
      *d = 1;
    if(n == 20)
      assert(b == 0);
    n++;
  }
  return 1;
}
