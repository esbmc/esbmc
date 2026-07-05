// Regression for #5313: enforcing a contract on a recursive function must use
// the function's own contract for the recursive self-call instead of unwinding
// the real recursion unboundedly (which previously ran out of memory). The
// recursion terminates in at most 4 calls, well within any unwind bound, and
// the contract holds, so verification must terminate with SUCCESSFUL.
int rec(int n)
{
  __ESBMC_requires(n >= 0);
  __ESBMC_requires(n < 4);
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_assigns();
  if (n == 0)
    return 0;
  return rec(n - 1);
}
