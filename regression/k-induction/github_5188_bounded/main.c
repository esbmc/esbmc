/* Companion to issue #5188: with the added value masked to a small range the
 * accumulation cannot overflow within a bounded unwind, so ESBMC verifies the
 * loop cleanly. Contrast with github_5188, where k-induction's unconstrained
 * inductive step leaves the accumulator and the added value unbounded, so the
 * same addition overflows. */
int g(int a, int b)
{
  int x = a, prod = 0;
  b = b & 0xFF; /* b in [0,255]: a bounded number of additions cannot overflow */
  while (x >= 0)
  {
    prod = prod + b;
    x--;
  }
  return prod;
}
