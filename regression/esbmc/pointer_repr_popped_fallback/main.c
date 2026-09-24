// A pointer rebuilt from memset bytes matches no flattened pointer, so it is
// the address-space reconstruction. --smt-symex-guard solves the branch in a
// pushed context; popping it must not drop that fallback.
#include <assert.h>
#include <string.h>

int nondet_int(void);

int main(void)
{
  int *p;
  memset(&p, 0, sizeof(p));
  int *q = p;
  int x = 0;
  if (nondet_int())
    x = 1;
  assert(q == 0);
  return x;
}
