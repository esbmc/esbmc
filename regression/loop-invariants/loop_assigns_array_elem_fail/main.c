/* The ASSERT half of the same path: a loop body writing an element its
 * __ESBMC_loop_assigns does not name must be reported, and reported by
 * element rather than by array. */
int global[8];

int main()
{
  int i = 0;

  __ESBMC_loop_assigns(i, global[3]);
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  while (i < 4)
  {
    global[3] = i;
    global[5] = i; /* not in __ESBMC_loop_assigns */
    i++;
  }

  return 0;
}
