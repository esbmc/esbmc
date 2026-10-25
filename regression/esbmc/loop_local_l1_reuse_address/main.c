int main()
{
  int r = 0;
  for (int i = 0; i < 2; i++)
  {
    int x = i;
    int *p = &x;
    r += *p;
  }
  __ESBMC_assert(r == 1, "each pointer reads its own iteration's x");
  return r;
}
