/* A replace-side __ESBMC_is_fresh must hold for the object the pointer names,
 * not for the syntax the call site happens to use. `buf` is the same stack
 * array in both calls; only the second reaches the callee through a parameter. */
void callee(int *p, unsigned n)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, n * sizeof(int)));
  __ESBMC_requires(n > 0);
  __ESBMC_assigns(p[0]);
  __ESBMC_ensures(p[0] == 1);
  p[0] = 1;
}

void mid(int *p, unsigned n)
{
  callee(p, n);
}

int main(void)
{
  int buf[8];
  callee(buf, 8);
  mid(buf, 8);
  return 0;
}
