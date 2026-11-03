// Pointer-through-function k-induction havoc regression.
//
// dispatch(&obj) mutates obj via pointer argument. The k-induction
// modified-variable analysis used to miss this (syntactic walker
// only tracked lhs of ASSIGN and callee-local pointers), causing
// I(k) to assume obj unchanged across the inductive step.
//
// Expected: VERIFICATION FAILED — assert(0) is reachable when all
// three choice branches execute to mutate obj.x / obj.y up to 5.

#include <assert.h>

struct S { int x; int y; };
struct S obj = {0, 0};
int nondet_int(void);

void dispatch(struct S *p)
{
  int choice = nondet_int();
  if (choice == 0) { if (p->x < 5) p->x++; }
  else if (choice == 1) { if (p->y < 5) p->y++; }
  else {
    __ESBMC_assume(p->x == 5);
    __ESBMC_assume(p->y == 5);
    assert(0);
  }
}

int main(void)
{
  while (nondet_int())
    dispatch(&obj);
  return 0;
}
