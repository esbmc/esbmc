// The loop moves d between a and b, so d is no root: the object it points to
// at the loop head is not the only one bump writes. Havocing that object
// alone would leave b->state at its pre-loop 0 and prove b->state < 5.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int state;
};

void bump(struct dev *d) { d->state = d->state + 1; }

int main()
{
  struct dev *a = malloc(sizeof(struct dev));
  struct dev *b = malloc(sizeof(struct dev));
  a->state = 0;
  b->state = 0;
  struct dev *d = a;
  for (;;)
  {
    bump(d);
    __VERIFIER_assert(b->state < 5);
    d = d == a ? b : a;
  }
}
