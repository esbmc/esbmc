int g;

void count(void)
{
  unsigned i = 0;
  __ESBMC_loop_invariant(i <= 3 && g == 0);
  for (; i < 3; i++)
    ;
}

int main(void)
{
  g = 1;
  return 0;
}
