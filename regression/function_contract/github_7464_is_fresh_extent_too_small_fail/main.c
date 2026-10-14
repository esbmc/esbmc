/* The forwarded case must still check the extent, not merely stop reporting a
 * free VALID_OBJECT. `buf` holds 4 ints; the callee's contract demands 8, so
 * the call site cannot discharge it. */
void callee(int *p, unsigned n)
{
  __ESBMC_requires(__ESBMC_is_fresh(p, n * sizeof(int)));
  __ESBMC_requires(n > 0);
  __ESBMC_assigns(p[0]);
  __ESBMC_ensures(p[0] == 1);
  p[0] = 1;
}

void mid(int *p)
{
  callee(p, 8);
}

int main(void)
{
  int buf[4];
  mid(buf);
  return 0;
}
