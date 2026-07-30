/* github_6212_frame_clause_pass:
 *   Companion to github_6212_frame_clause_fail. Once the contract states the
 *   extent, the witness bound is justified and assigns compliance is checked
 *   as normal.
 */
void f(int *a, int i)
{
  __ESBMC_requires(a != 0);
  __ESBMC_requires(__ESBMC_is_fresh(a, 4 * sizeof(int)));
  __ESBMC_requires(i == 0);
  __ESBMC_assigns(a[i]);
  __ESBMC_ensures(1);
  a[i] = 1;
}

int main()
{
  return 0;
}
