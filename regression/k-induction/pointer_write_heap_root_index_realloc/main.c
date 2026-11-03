// realloc copies only an index into d's new object, so what the loop loads
// from it is an offset into that object. The inductive step havocs d's object
// whole and proves calls < 3.
#include <stdlib.h>
extern void abort(void);
extern int nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int calls;

int main()
{
  long *d0 = malloc(4 * sizeof(long));
  d0[0] = 1;
  long *d = realloc(d0, 4 * sizeof(long));
  for (;;)
  {
    long n = d[0];
    if (n > 0 && n < 4)
      d[n] = nondet_int();
    calls = (calls + 1) % 3;
    __VERIFIER_assert(calls < 3);
  }
}
