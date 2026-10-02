// probe increments d->state through an integer derived from d. Unless the
// inductive step havocs the whole object d points to, d->state keeps its
// pre-loop 0 and the step proves d->state < 5.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int state;
};

void *alloc(unsigned n) { return malloc(n); }

void probe(struct dev *d)
{
  unsigned long base = (unsigned long)d;
  *(int *)(base + 0) = d->state + 1;
}

int main()
{
  struct dev *d = alloc(sizeof(struct dev));
  d->state = 0;
  for (;;)
  {
    probe(d);
    __VERIFIER_assert(d->state < 5);
  }
}
