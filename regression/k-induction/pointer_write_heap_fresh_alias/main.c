// KNOWNBUG: the inductive step no longer skips objects a run-once loop
// allocates, which was unsound (pointer_write_heap_fresh_kept_fail).
// p holds &x or an object the loop allocates. The inductive step havocs x,
// pins p to x or an unresolved address, and proves y, which no write
// reaches.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int x, y;

int main()
{
  int *p = &x;
  int i = 0;
  for (;;)
  {
    x = 0;
    if (i >= 5)
      *p = 5;
    __VERIFIER_assert(y == 0);
    if (i == 5)
      p = malloc(sizeof(int));
    i++;
  }
}
