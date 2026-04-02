/* replace_forall_requires_pass:
 * __ESBMC_forall in requires clause, caller satisfies it.
 * --replace-call-with-contract should: ASSERT requires (passes), ASSUME ensures.
 * Final assert on return value must hold via the ensured postcondition.
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
  int a[N], n, i;
  __ESBMC_assume(n > 0 && n <= N);
  /* satisfy the forall requires */
  __ESBMC_assume(
    __ESBMC_forall(&i, !(i >= 0 && i < N) || (a[i] >= -100 && a[i] <= 100)));
  int r = find_min(a, n);
  assert(r >= -100 && r <= 100);
  return 0;
}
