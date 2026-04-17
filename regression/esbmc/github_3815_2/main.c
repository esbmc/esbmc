/* Regression test for issue #3815:
 * Singly-recursive foo_rec with --unwind 3 verifies foo_rec(n) == foo_iter(n)
 * for n < 3.  --unwind 3 is sufficient because the maximum recursion depth for
 * n < 3 is 2 (foo_rec(2) -> foo_rec(1) -> foo_rec(0) -> base case).
 *
 * Singly-recursive functions require O(N) inlinings for --unwind N,
 * so even large values are tractable.
 */

unsigned int foo_rec(const unsigned int n)
{
  if (n == 0)
    return 0;
  return n + foo_rec(n - 1);
}

unsigned int foo_iter(const unsigned int n)
{
  unsigned int sum = 0;
  for (unsigned int i = 0; i <= n; ++i)
    sum += i;
  return sum;
}

int main()
{
  unsigned int n;
  __ESBMC_assume(n < 3);
  unsigned int rec = foo_rec(n);
  unsigned int iter = foo_iter(n);
  __ESBMC_assert(rec == iter, "foo_rec matches foo_iter");
  return 0;
}
