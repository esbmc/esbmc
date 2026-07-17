int myabs(int a)
{
  if (a < 0)
    return -a;
  return a;
}

int main()
{
  int x;
  __ESBMC_assume(__ESBMC_forall(&x, myabs(x) >= 0));
  __ESBMC_assert(0, "unreachable when the assume is mismodelled");
}
