/* A global array with an element target used to be snapshotted as one scalar
 * and asserted unchanged, which the write the clause itself names falsifies.
 * Phase 2B only ever saw `p + i` from a pointer parameter; `global[i]` is an
 * index and reached neither that nor the per-field machinery. */
int global[10];

void write_global_elem(int i, int v)
{
  __ESBMC_requires(i >= 0 && i < 10);
  __ESBMC_assigns(global[i]);
  __ESBMC_ensures(1);
  global[i] = v; /* only touches global[i], which is in assigns */
}

int main()
{
  return 0;
}
