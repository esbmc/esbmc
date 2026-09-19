// The loop writes a heap object through p, which it never moves: unless the
// inductive step havocs *p, *p keeps its pre-loop value 0 and the step proves
// *p < 10.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int *p = malloc(sizeof(int));
  int i = 0;
  *p = 0;
  for (;;)
  {
    *p = *p + 1;
    i++;
    __VERIFIER_assert(*p < 10);
  }
}
