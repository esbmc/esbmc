/* Companion to issue #5188: the same loop verifies cleanly under a bounded
 * unwind (no k-induction), confirming the overflow report in github_5188 is
 * specific to k-induction's unconstrained inductive step. */
int g(int a, int b)
{
  int x = a, prod = 0;
  while (x >= 0)
  {
    prod = prod + b;
    x--;
  }
  return prod;
}
