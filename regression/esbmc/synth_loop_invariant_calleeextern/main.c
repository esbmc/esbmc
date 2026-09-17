/* A dependency with no body -- here the undeclared-elsewhere `opaque` -- gives
 * the call-graph closure nothing to walk. That is not a reason to stand down:
 * a function ESBMC has no body for has no loop to cut. count_to is still
 * protected, and the run still proves the user's invariant. */
unsigned int opaque(unsigned int n);

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

int main(void)
{
  unsigned int i = 0;
  unsigned int seed = opaque(4);

  __ESBMC_loop_invariant(i <= 4);
  __ESBMC_loop_invariant(count_to(0) == 0);
  while (i < 4)
    i++;

  __ESBMC_assert(i == 4, "end");
  return (int)seed;
}
