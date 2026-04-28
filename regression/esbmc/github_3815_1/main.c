/* Regression test for issue #3815:
 * Doubly-recursive fib_rec with --unwind 3 verifies fib_rec(n) == fib_iter(n)
 * for n < 3.  --unwind 3 is sufficient because the maximum recursion depth for
 * n < 3 is 2 (fib_rec(2) -> fib_rec(1) -> base case).
 *
 * Note: k-way recursive functions with --unwind N require O(k^N) inlinings.
 * Use --unwind values just large enough to cover the actual recursion depth.
 */

unsigned int fib_rec(const unsigned int n)
{
  if (n == 0)
    return 0;
  else if (n == 1)
    return 1;
  else
    return fib_rec(n - 1) + fib_rec(n - 2);
}

unsigned int fib_iter(const unsigned int n)
{
  if (n == 0)
    return 0;
  if (n == 1)
    return 1;
  unsigned int a = 0, b = 1;
  for (unsigned int i = 2; i <= n; ++i)
  {
    unsigned int t = a + b;
    a = b;
    b = t;
  }
  return b;
}

int main()
{
  unsigned int n;
  __ESBMC_assume(n < 3);
  unsigned int rec = fib_rec(n);
  unsigned int iter = fib_iter(n);
  __ESBMC_assert(rec == iter, "fib_rec matches fib_iter");
  return 0;
}
