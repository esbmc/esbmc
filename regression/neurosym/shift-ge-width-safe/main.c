/* SMT-LIB2 FixedSizeBitVectors folds a shift by >= the operand width to
 * zero. The shift amount here reaches the backend's local model evaluator
 * straight from NeuroSym's own model output, where evaluating it as a plain
 * BigInt shift instead builds 2^y through a y-iteration multiply loop and
 * does not terminate. Wrapping the result in a struct is what makes the
 * trace query the shift as a composite expression rather than a symbol. */
struct wrap
{
  unsigned int f;
};

int main()
{
  unsigned int x = nondet_uint();
  unsigned int y = nondet_uint();
  struct wrap w;

  /* Pinned, not merely >= 32: an unguarded evaluator computes 2^y through
   * a y-iteration BigInt multiply loop, so only a shift amount this large
   * makes the missing guard exceed the test timeout. A solver left free to
   * pick any y >= 32 picks a small one, and the test stops biting. */
  __ESBMC_assume(y == 100000000u);
  w.f = x << y;
  __ESBMC_assert(w.f == 0u, "shift by at least the width is zero");
  return 0;
}
