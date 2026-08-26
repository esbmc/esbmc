/* github #7184: each array named by the clause keeps its own reported property,
 * so a violation on b is not masked by a's frame check passing. */
void f(int *a, int *b, int n, int i, int k)
{
  __ESBMC_requires(__ESBMC_is_fresh(a, n * sizeof(int)));
  __ESBMC_requires(__ESBMC_is_fresh(b, n * sizeof(int)));
  __ESBMC_requires(n >= 3);
  __ESBMC_requires(i == 0 && k == 0);
  __ESBMC_assigns(a[i], b[k]);

  a[i] = 1;
  b[k] = 2;
  b[2] = 3;
}

int main()
{
  return 0;
}
