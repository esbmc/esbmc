// Tests that --interval-analysis tightens the inductive hypothesis when
// combined with --k-induction. Interval analysis derives 0 <= i <= 10 at
// the loop head; the post-k-induction pass assumes that bound right after
// k-induction's nondet havoc, so the inductive step can discharge the
// post-loop assertion.
int main()
{
  unsigned int i = 0;
  unsigned int sum = 0;
  while (i < 10)
  {
    sum += 1;
    i++;
  }
  // After the loop: i == 10 and sum == 10.
  __ESBMC_assert(sum == 10, "sum equals iteration count");
  return 0;
}
