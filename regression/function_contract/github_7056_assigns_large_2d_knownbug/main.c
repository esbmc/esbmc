/* The store form is not used when the element is itself an array: reading the
 * array-typed rvalue `m[i]` back out of the snapshot fails on every solver,
 * the same gap that stops __ESBMC_old over an array (#7057). So a
 * two-dimensional global past the element-wise cap keeps the whole-array
 * assertion, which the write the clause itself names falsifies.
 *
 * Below the cap the element-wise form handles this shape
 * (github_7056_assigns_global_2d); only the large case is uncovered. */
int m[300][4];

void f(int i, int v)
{
  __ESBMC_requires(i >= 0 && i < 300);
  __ESBMC_assigns(m[i]);
  __ESBMC_ensures(1);
  m[i][0] = v; /* only touches m[i], which is in assigns */
}

int main()
{
  return 0;
}
