// main also calls make before the loop, so make's objects can predate it:
// the loop writes keep's object through last. Leaving them unhavoced would
// let the inductive step prove *keep == 0, so the step is disabled and the
// base case finds the bug.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int *keep, *last;
int *make(void) { return malloc(sizeof(int)); }

void run(void)
{
  int c = 0;
  do
  {
    if (keep && c >= 5)
      __VERIFIER_assert(*keep == 0);
    if (c == 0 && last)
      *last = 1;
    int *n = make();
    *n = 0;
    last = n;
    c++;
  } while (__VERIFIER_nondet_int());
}

int main()
{
  keep = make();
  *keep = 0;
  last = keep;
  run();
}
