/* github_6212_frame_clause_fail:
 *   Adding a frame clause must not restore the #6212 hole. Phase 2B bounds a
 *   nondet witness index j by the array's extent so that arr[j] is a valid
 *   read; over a parameter whose extent the contract never states, that bound
 *   would assume the extent is at least one element, which is exactly the
 *   assumption #6212 removes. Such parameters are skipped by Phase 2B instead,
 *   so a[0] stays unjustified here and the write is caught.
 *
 *   github_6212_frame_clause_pass is the same contract with the extent stated.
 */
void f(int *a, int i)
{
  __ESBMC_requires(a != 0);
  __ESBMC_requires(i == 0);
  __ESBMC_assigns(a[i]);
  __ESBMC_ensures(1);
  a[i] = 1;
}

int main()
{
  return 0;
}
