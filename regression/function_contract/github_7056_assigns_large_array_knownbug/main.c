/* One assertion per element costs more than linearly -- 256 elements solve in
 * 0.6s, 512 in 2.4s, 1000 in 16s, 10000 not at all -- so past a capped extent
 * the whole array is asserted unchanged instead. That is sound but imprecise:
 * the write the clause itself names falsifies it, so this correct body is
 * reported. Encoding the spared element as a quantifier rather than an
 * assertion per element would lift the cap. */
int global[512];

void f(int i, int v)
{
  __ESBMC_requires(i >= 0 && i < 512);
  __ESBMC_assigns(global[i]);
  __ESBMC_ensures(1);
  global[i] = v; /* only touches global[i], which is in assigns */
}

int main()
{
  return 0;
}
