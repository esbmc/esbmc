/* Regression test: modular verification with enforce + replace + loop invariants
 *
 * Step 1 (enforce): verify count_to satisfies its contract using loop invariant.
 * Step 2 (replace): verify main using count_to's contract summary.
 *
 * Both steps must succeed independently.
 * Run enforce:  --enforce-contract count_to --loop-invariant
 * Run replace:  --replace-call-with-contract count_to --loop-invariant
 */
extern int __VERIFIER_nondet_int(void);

int count_to(int n)
{
  __ESBMC_requires(n >= 0 && n <= 10);
  __ESBMC_ensures(__ESBMC_return_value == n);

  int i = 0;
  __ESBMC_loop_invariant(i >= 0 && i <= n);
  while(i < n)
    i++;
  return i;
}

int main()
{
  int n = __VERIFIER_nondet_int();
  __ESBMC_assume(n >= 0 && n <= 10);

  int k = count_to(n); /* after replace: k == n (nondet satisfying ensures) */

  int j = 0;
  __ESBMC_loop_invariant(j >= 0 && j <= k);
  while(j < k)
    j++;
  assert(j == k);
  return 0;
}
