/* The body also writes global[5], outside the clause. */
int global[512];

int main()
{
  int i = 0;

  __ESBMC_loop_assigns(i, global[3]);
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  while (i < 4)
  {
    global[3] = 1;
    global[5] = 2;
    i++;
  }
  return 0;
}
