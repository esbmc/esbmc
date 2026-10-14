// Issue #5230 (k-induction Phase 2) soundness gate: a loop writes an array
// element through a pointer to a *heap* object (`dest = malloc(...)`). The
// value-set fixpoint resolves the written pointer `dest` to a dynamic
// (heap) object, which has no nameable symbol to havoc. Phase 2 must NOT
// resolve it — doing so would be the unsound generalisation that #5027
// documents — so it abstains and falls back to Phase 1: the pointer-write
// warning is emitted and the inductive step is disabled. This safe program
// is then proven by the forward condition.
#include <stdlib.h>
extern unsigned char nondet_uchar(void);

int main(void)
{
  unsigned char (*dest)[8] = malloc(8); // dest points to a heap object
  __ESBMC_assume(dest != 0);

  for (int i = 0; i < 8; i++)
    (*dest)[i] = nondet_uchar(); // index through a pointer to a dynamic object

  return 0;
}
