#include <stdatomic.h>

/* The call is a sideeffect2t in a loop condition, which carries no location of
 * its own. Unless the enclosing statement's is picked up, the counterexample
 * names no file and reports line 0. */
int main(void)
{
  atomic_int *p = 0;
  while (atomic_load(p) < 10)
    ;
  return 0;
}
