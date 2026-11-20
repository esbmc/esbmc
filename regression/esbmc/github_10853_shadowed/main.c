int g;
int main(void)
{
  __ESBMC_loop_invariant(g == 0);
  int g = 5;
  while (g < 10)
    g++;
  return 0;
}
