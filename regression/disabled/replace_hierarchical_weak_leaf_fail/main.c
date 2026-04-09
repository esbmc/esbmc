/*
 * Test: Full contract replacement trusts all contracts (even weak ones)
 * Expected: VERIFICATION SUCCESSFUL
 *
 * Scenario:
 *   - leaf_double has a WEAK contract: ensures(ret >= x), actual body returns 2*x
 *   - parent_compute's contract claims: ensures(ret >= x * 2 + 1)
 *
 * With --replace-call-with-contract "*":
 *   - ALL functions are replaced with their contracts
 *   - parent_compute is replaced: assume(r >= 5*2+1 = 11) → assert(r >= 11) → SUCCESS
 *   - The weak leaf contract is irrelevant because parent's body is never executed
 *
 * To verify that parent_compute's body actually satisfies its contract
 * (using the weak leaf contract), use --enforce-contract parent_compute.
 */
#include <assert.h>
int leaf_double(int x)
{
  __ESBMC_requires(x >= 0 && x <= 50);
  // WEAK contract: only guarantees ret >= x, not ret == 2*x
  __ESBMC_ensures(__ESBMC_return_value >= x);
  return x * 2;
}
int parent_compute(int x)
{
  __ESBMC_requires(x >= 0 && x <= 50);
  __ESBMC_ensures(__ESBMC_return_value >= x * 2 + 1);
  int doubled = leaf_double(x);  // body: 2*x, but contract only: >= x
  int result = doubled + 1;
  return result;
}
int main()
{
  int a = 5;
  int r = parent_compute(a);
  // parent_compute's contract says r >= 11 (= 5*2+1)
  // But with weak leaf contract, ESBMC only knows doubled >= 5
  // So r = doubled + 1 >= 6, which does NOT guarantee r >= 11
  assert(r >= 11);
  return 0;
}
