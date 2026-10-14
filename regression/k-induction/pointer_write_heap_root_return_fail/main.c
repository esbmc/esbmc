// get returns its argument, so the pointer bump writes through is d itself.
// Unless the inductive step follows the call to d and havocs its object,
// d->state keeps its pre-loop 0 and the step proves d->state < 5.
#include <stdlib.h>
extern void abort(void);
void reach_error() {}
void __VERIFIER_assert(int c) { if (!c) { ERROR: { reach_error(); abort(); } } }

struct dev
{
  int state;
};

struct dev *get(struct dev *d) { return d; }

void bump(struct dev *d)
{
  struct dev *e = get(d);
  e->state = e->state + 1;
}

int main()
{
  struct dev *d = malloc(sizeof(struct dev));
  d->state = 0;
  for (;;)
  {
    bump(d);
    __VERIFIER_assert(d->state < 5);
  }
}
