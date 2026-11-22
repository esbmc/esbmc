// The inner loop increments a's object through the temporary that holds d
// before d++. The outer loop's inductive step must havoc that object, so it
// cannot prove a[0] < 3, and the base case finds the bug.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  unsigned char *a = malloc(4);
  a[0] = 0;
  unsigned char *d = a;
  for (;;)
  {
    int once = 1;
    while (once)
    {
      *d++ += 1;
      d--;
      once = 0;
    }
    __VERIFIER_assert(a[0] < 3);
  }
}
