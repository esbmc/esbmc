#include <assert.h>

// Exercises the shl constant-fold guard in expr_simplifier.cpp: a shift by a
// negative constant amount must not be folded on uint64_t (a signed -1 would
// become a 2^64-1 shift, which is undefined behaviour). The fold is declined
// and the shift is left for the solver, while normal folds still apply.
int main()
{
  int amt = 2 - 3;           // -1, a negative constant shift amount
  volatile int r = 1 << amt; // out-of-range amount: no internal UB fold
  (void)r;

  int ok = 4 << 1; // in-range amount still folds normally
  assert(ok == 8);
  return 0;
}
