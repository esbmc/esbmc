/* The body writes only the granted element of an array past the cap. */
int global[512];

int main()
{
  int i = 0;

  __ESBMC_loop_assigns(i, global[3]);
  __ESBMC_loop_invariant(i >= 0 && i <= 4);
  while (i < 4)
  {
    global[3] = 1;
    i++;
  }
  return 0;
}
