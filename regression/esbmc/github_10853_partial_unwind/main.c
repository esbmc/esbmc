int main(void)
{
  unsigned k = 0;
  while (k < 5)
  {
    unsigned j = 0;
    __ESBMC_loop_invariant(j <= 1 && k < 2);
    for (; j < 1; j++)
      ;
    k++;
  }
  return 0;
}
