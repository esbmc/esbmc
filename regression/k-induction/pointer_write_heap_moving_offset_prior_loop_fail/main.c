// The first loop moves s but writes nothing through it; the second increments
// a's object through s. The first loop's havoc must keep s inside a's object,
// or the second loop's writes through s reach nothing and its inductive step
// proves a[0] < 3.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int *a = malloc(4 * sizeof(int));
  a[0] = 0;
  int *s = a;
  while (__VERIFIER_nondet_int())
  {
    s++;
    s--;
  }
  for (;;)
  {
    *s = *s + 1;
    __VERIFIER_assert(a[0] < 3);
  }
}
