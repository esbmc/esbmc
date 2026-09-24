// A pointer rebuilt from memset bytes stays NULL after --smt-symex-guard
// pops the branch context, so asserting it is not NULL must fail.
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
  assert(q != 0);
  return x;
}
