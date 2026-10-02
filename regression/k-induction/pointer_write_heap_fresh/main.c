// The loop writes, through a pointer it moves, objects that make allocates
// on every iteration. run executes once and only the loop calls make, so no
// such object exists before the loop and the inductive step needs no havoc
// for them: keep's object, which the loop never writes, stays 0.
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
  keep = malloc(sizeof(int));
  *keep = 0;
  run();
}
