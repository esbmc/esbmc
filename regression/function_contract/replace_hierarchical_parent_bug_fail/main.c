/*
 * Test: Full contract replacement trusts all contracts
 * Expected: VERIFICATION SUCCESSFUL
 *
 * Scenario:
 *   - parent_func has a BUG in its body (subtracts 5 instead of adding 5)
 *   - But parent_func's contract claims: ensures(ret == x + 15)
 *
 * With --replace-call-with-contract "*":
 *   - ALL functions (including parent_func) are replaced with their contracts
 *   - The body bug is not visible — replace trusts the contract
 *   - main() sees: assume(r == 3 + 15 = 18) → assert(r == 18) → SUCCESS
 *
 * To catch the body bug, use --enforce-contract parent_func instead.
 * Replace mode = "trust the contract", Enforce mode = "verify the contract".
 */
#include <assert.h>
int leaf_add(int x)
{
  __ESBMC_requires(x >= 0 && x <= 100);
  __ESBMC_ensures(__ESBMC_return_value == x + 10);
  return x + 10;
}
int parent_func(int x)
{
  __ESBMC_requires(x >= 0 && x <= 100);
  __ESBMC_ensures(__ESBMC_return_value == x + 15);
  int intermediate = leaf_add(x);  // gets x + 10
  // BUG: should be intermediate + 5, but we wrote - 5
  int result = intermediate - 5;   // actual: x + 5, not x + 15
  return result;
}
int main()
{
  int a = 3;
  int r = parent_func(a);
  // Per parent_func's contract, r should be 3 + 15 = 18
  // But actual body computes 3 + 10 - 5 = 8
  // Hierarchical replacement keeps parent body → catches this bug
  assert(r == 18);
  return 0;
}
