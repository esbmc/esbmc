// probe writes the object d points to through an integer derived from its
// parameter, and alloc is called before the loop too, so the points-to sets
// cannot tell its objects apart. d never moves, so the inductive step havocs
// its whole object and proves the property. b, which probe allocates, needs
// no havoc: it is allocated in the iteration that writes it.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int state;
  int *buf;
};

int calls;

void *alloc(unsigned n) { return malloc(n); }

void probe(struct dev *d)
{
  unsigned long base = (unsigned long)d;
  *(int *)(base + 0) = 1;
  int *b = alloc(sizeof(int));
  *b = 0;
  d->buf = b;
  calls = (calls + 1) % 3;
}

int main()
{
  struct dev *d = alloc(sizeof(struct dev));
  d->state = 0;
  for (;;)
  {
    probe(d);
    __VERIFIER_assert(calls < 3);
  }
}
