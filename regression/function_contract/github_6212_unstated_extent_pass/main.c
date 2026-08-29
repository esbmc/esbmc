/* github_6212_unstated_extent_pass:
 * Same body as github_6212_unstated_extent_fail, but the contract now states
 * the extent, so p[20] is justified and the write verifies. The companion
 * test checks that dropping the is_fresh makes the same write fail.
 */
void f(int *p)
{
  __ESBMC_requires(p != 0);
  __ESBMC_requires(__ESBMC_is_fresh(p, 21 * sizeof(int)));
  __ESBMC_ensures(1);
  p[20] = 1;
}

int main()
{
  return 0;
}
