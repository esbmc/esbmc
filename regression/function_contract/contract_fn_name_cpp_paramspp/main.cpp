int fst(int x)
{
  __ESBMC_ensures(__ESBMC_return_value >= 0);
  return x > 0 ? x : 0;
}

int main()
{
  int b = fst(3);
  __ESBMC_assert(b == 3, "true of the body, not of the contract");
  return 0;
}
