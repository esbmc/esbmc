// The loop increments a's object through s, which it moves only by offsets of
// itself. The inductive step must havoc that object, not just move s, or a[0]
// keeps its pre-loop 0 and the step proves a[0] < 3.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  unsigned char *a = malloc(4);
  a[0] = 0;
  unsigned char *s = a;
  for (;;)
  {
    *s = *s + 1;
    if (__VERIFIER_nondet_int())
    {
      s++;
      s--;
    }
    __VERIFIER_assert(a[0] < 3);
  }
}
