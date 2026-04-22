/* Test: contract-annotated C files can be verified with direct BMC.
 * Previously, __ESBMC_requires/__ESBMC_ensures were only defined when
 * --enforce-contract was active, so annotated files would crash with
 * "Function call to non-intrinsic prefixed with __ESBMC" without that flag.
 * Now all contract macros are always available: requires/ensures become
 * ASSUME instructions and assigns/old are no-ops in direct BMC mode.
 */
int add(int x, int y)
{
  __ESBMC_requires(x >= 0 && y >= 0);
  __ESBMC_assigns(x);
  return x + y;
}

int main()
{
  int r = add(3, 4);
  __ESBMC_assert(r == 7, "3+4=7");
  return 0;
}
