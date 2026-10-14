// Companion to github_5313_recursive_enforce_pass: enforcing a recursive
// contract must still catch a contract violation. Here the recursive call
// rec(n + 1) can reach n + 1 == 4, which violates the callee's own
// precondition (n < 4). Because the self-call is replaced by the contract,
// its requires becomes a proof obligation at the call site, so verification
// must report FAILED (and would, before the fix, instead unwind unboundedly).
int rec(int n)
{
  __ESBMC_requires(n >= 0);
  __ESBMC_requires(n < 4);
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  __ESBMC_assigns();
  if (n == 0)
    return 0;
  return rec(n + 1);
}
