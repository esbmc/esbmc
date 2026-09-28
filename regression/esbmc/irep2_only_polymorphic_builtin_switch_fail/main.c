#include <stdatomic.h>

/* A switch selector is a call site too: the sideeffect2t holding the call
 * carries no location, so the counterexample takes the switch statement's. */
int main(void)
{
  atomic_int *p = 0;
  switch (atomic_load(p))
  {
  case 1:
    return 1;
  default:
    return 0;
  }
}
