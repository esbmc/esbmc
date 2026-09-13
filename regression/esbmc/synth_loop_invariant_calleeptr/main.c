/* The user's invariant calls through a function pointer, so the callee whose
 * loops must be left alone cannot be named. Synthesis stands down for the whole
 * program rather than guess which function the marker depends on. */
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

int main(void)
{
  unsigned int i = 0;

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant(fp(4) == 4);
  while (i < 4)
    i++;

  __ESBMC_assert(i == 4, "end");
  return 0;
}
