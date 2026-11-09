// The loop writes through p, which holds what id returns. goto_convert keeps
// that value in a temporary, which the trace follows to h, a pointer the loop
// never moves: the inductive step havocs h's object and proves *h == 5.
#include <stdlib.h>
extern void abort(void);
extern int __VERIFIER_nondet_int(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

int *id(int *q) { return q; }

int main()
{
  int *h = malloc(sizeof(int));
  *h = 0;
  while (__VERIFIER_nondet_int())
  {
    int *p = id(h);
    *p = 5;
    __VERIFIER_assert(*h == 5);
  }
}
