int myabs(int a)
{
  if (a < 0)
    return -a;
  return a;
}

int main()
{
  int x;
  __ESBMC_assert(__ESBMC_forall(&x, myabs(x) >= 0), "fails at INT_MIN");
}
