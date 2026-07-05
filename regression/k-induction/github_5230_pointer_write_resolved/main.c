// Issue #5230 (k-induction Phase 2): a loop writes an array element through a
// pointer (`(*dest)[i % 8]`), the same shape that #5224 had to gate. Phase 2
// resolves the written pointer `dest` against the value-set fixpoint to the
// single named object `a` and havocs `a` as a whole symbol, so the inductive
// step stays enabled and sound instead of being disabled outright.
//
// The loop counts are unbounded, so the forward condition cannot close the
// proof; only the inductive step can. The property under test is the scalar
// invariant `x == 0 && y == 0`, which the inductive step proves once `a` is
// havoc'd. Under Phase 1 the array write disabled the inductive step program-
// wide and this returned VERIFICATION UNKNOWN; Phase 2 proves it.
//
// The plain `y` loop comes BEFORE the pointer-array loop: it is transformed
// first, so the value-set fixpoint must be built up front (before any loop is
// transformed), not lazily once the pointer-array loop is reached — building
// it on a partially-transformed CFG aborts.
extern unsigned char nondet_uchar(void);
extern int nondet_int(void);

int main(void)
{
  unsigned char a[8];
  unsigned char (*dest)[8] = &a;

  int n = nondet_int();
  __ESBMC_assume(n > 0);

  int y = 0;
  for (int j = 0; j < n; j++)
    y = 0; // plain loop, transformed before the pointer-array loop

  int x = 0;
  for (int i = 0; i < n; i++)
  {
    (*dest)[i % 8] = nondet_uchar(); // resolved to `a`, havoc'd as a symbol
    x = 0;                           // scalar inductive invariant
  }

  __ESBMC_assert(x == 0 && y == 0, "scalars stay zero across all iterations");
  return 0;
}
