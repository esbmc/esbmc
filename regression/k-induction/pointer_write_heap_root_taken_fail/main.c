// The loop moves d through its address, so d is no root. Havocing only the
// object d points to at the loop head would leave spare->state at its
// pre-loop 0 and prove spare->state < 5.
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
  struct dev *d = malloc(sizeof(struct dev));
  struct dev *spare = malloc(sizeof(struct dev));
  d->state = 0;
  spare->state = 0;
  struct dev **pd = &d;
  for (;;)
  {
    bump(d);
    __VERIFIER_assert(spare->state < 5);
    if (d->state == 3)
      *pd = spare;
  }
}
