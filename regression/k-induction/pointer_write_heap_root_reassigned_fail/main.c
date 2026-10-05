// The loop increments d->state through integers it computes from d in every
// iteration. Unless the inductive step havocs the whole object d points to,
// d->state keeps its pre-loop 0 and the step proves d->state < 5.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int state;
};

void *alloc(unsigned n) { return malloc(n); }

int main()
{
  alloc(1);
  struct dev *d = alloc(sizeof(struct dev));
  d->state = 0;
  for (;;)
  {
    unsigned long base = (unsigned long)d;
    unsigned long field = base + 0;
    *(int *)field = d->state + 1;
    __VERIFIER_assert(d->state < 5);
  }
}
