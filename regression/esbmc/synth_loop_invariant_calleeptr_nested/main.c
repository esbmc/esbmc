/* The marker's own call is direct, but the callee reaches an indirect one, so
 * closing over the call graph still runs out of names. The whole-program stand
 * down covers that case too, not only an indirect call in the marker block. */
static unsigned int count_to(unsigned int n)
{
  unsigned int i = 0;
  unsigned int s = 0;
  while (i < n)
  {
    s = s + 1;
    i = i + 1;
  }
  return s;
}

static unsigned int (*fp)(unsigned int) = count_to;

static unsigned int outer(unsigned int n)
{
  return fp(n);
}

int main(void)
{
  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant(outer(4) == 4);
  while (i < 4)
    i++;

  __ESBMC_assert(i == 4, "end");
  return 0;
}
