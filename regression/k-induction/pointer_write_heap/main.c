// The loop writes a heap object through p, which it never moves, so the
// inductive step havocs *p and still proves the property.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int main()
{
  int *p = malloc(sizeof(int));
  *p = 0;
  for (;;)
  {
    *p = (*p + 1) % 10;
    __VERIFIER_assert(*p < 10);
  }
}
