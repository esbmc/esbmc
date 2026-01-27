/*
 * Test: Hierarchical replacement soundness — weak leaf contract
 * Expected: VERIFICATION FAILED
 *
 * Scenario:
 *   - leaf_double(x) body returns x * 2, but contract is WEAK:
 *     ensures(__ESBMC_return_value >= x) — only says "at least x"
 *   - parent_compute(x) calls leaf_double(x) and adds 1 to the result.
 *     Contract claims: ensures(__ESBMC_return_value >= x * 2 + 1)
 *     Body actually does: leaf_double(x) + 1 = x*2 + 1 (correct body).
 *   - main() calls parent_compute(5) and asserts result >= 11 (= 5*2+1).
 *
 * With hierarchical replacement (--replace-call-with-contract "*"):
 *   - leaf_double is a "leaf" → replaced with weak contract (ret >= x)
 *   - parent_compute is a "parent" → body KEPT, leaf_double call replaced
 *   - When ESBMC inlines parent_compute's body:
 *       leaf_double(5) → havoc ret1; assume(ret1 >= 5)
 *       result = ret1 + 1
 *       The solver can pick ret1 = 5 (satisfies ret1 >= 5)
 *       → result = 6, which does NOT satisfy result >= 11
 *       → assert(result >= 11) can FAIL ✓
 *
 * This shows that a weak leaf contract causes the parent's body to
 * produce weaker guarantees, and the assertion can be violated.
 * The tool correctly reports VERIFICATION FAILED — it does not
 * unsoundly claim success.
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
