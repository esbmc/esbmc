/* replace_forall_requires_fail:
 * __ESBMC_forall in requires clause, caller does NOT satisfy it
 * (array elements may exceed [-100,100]).
 * --replace-call-with-contract should ASSERT requires -> VERIFICATION FAILED.
 */
#include <assert.h>
#define N 10

int find_min(int *a, int n)
{
  int i;
  __ESBMC_requires(a != ((void *)0));
  __ESBMC_requires(n > 0 && n <= N);
  __ESBMC_requires(
    __ESBMC_forall(&i, !(i >= 0 && i < n) || (a[i] >= -100 && a[i] <= 100)));
  __ESBMC_ensures(__ESBMC_return_value >= -100);
  __ESBMC_ensures(__ESBMC_return_value <= 100);
  int m = a[0];
  for(int j = 1; j < n; j++)
    if(a[j] < m)
      m = a[j];
  return m;
}

int main(void)
{
  int a[N], n;
  __ESBMC_assume(n > 0 && n <= N);
  /* array elements are unconstrained -- forall requires is NOT satisfied */
  int r = find_min(a, n);
  return 0;
}
