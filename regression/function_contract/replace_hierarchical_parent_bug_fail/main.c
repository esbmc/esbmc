/*
 * Test: Hierarchical replacement soundness — bug in parent function body
 * Expected: VERIFICATION FAILED
 *
 * Scenario:
 *   - leaf_add(x) has contract: ensures(__ESBMC_return_value == x + 10)
 *     Body correctly returns x + 10.
 *   - parent_func(x) calls leaf_add(x), then introduces a BUG:
 *     it subtracts 5 instead of adding 5. So actual result = (x+10) - 5 = x+5.
 *     But parent_func's contract claims: ensures(__ESBMC_return_value == x + 15).
 *   - main() calls parent_func(3) and asserts result == 18 (per parent's contract).
 *
 * With hierarchical replacement (--replace-call-with-contract "*"):
 *   - leaf_add is a "leaf" → replaced with contract (havoc + assume ret == x+10)
 *   - parent_func is a "parent" → body KEPT, internal leaf_add call replaced
 *   - When ESBMC inlines parent_func's body:
 *       leaf_add(3) → assume ret1 == 13
 *       result = ret1 - 5 = 8   (BUG! should be ret1 + 5 = 18)
 *       assert(result == 18) → FAILS ✓
 *
 * This proves that hierarchical replacement catches bugs in parent function
 * bodies, unlike naive full replacement which would just assume the
 * (incorrect) parent contract and miss the bug entirely.
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
