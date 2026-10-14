/* Soundness: the IS-UNSAT-as-non-termination signal must not fire
 * when k-induction's havoc transform was unreliable on the loop
 * driving the program's termination.
 *
 * Here the loop advances `g` through a function pointer. goto_loops's
 * call-graph analysis doesn't follow function pointers, so the loop's
 * modified set is empty and goto_k_inductiont skips the havoc.
 * Without the disable, IS at k=2 runs from the concrete initial state
 * `g = 0`, can't reach end-of-main in 2 iters, returns UNSAT, and
 * the strategy reports a false-positive FAILED. With the disable,
 * symex hits the function pointer, sets disable-inductive-step, and
 * the strategy treats the IS verdict as inconclusive — falling
 * through to FC, which closes at k=11. */
int g = 0;
void inc(void)
{
  g++;
}
int main(void)
{
  void (*fp)(void) = inc;
  while (g < 10)
    fp();
  return 0;
}
