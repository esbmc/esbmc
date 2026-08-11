int g = 0;
int bump(void) { g++; return g; }

int f(int x)
{
  __ESBMC_requires(bump() > 0);
  __ESBMC_ensures(__ESBMC_return_value == x);
  return x;
}

int main(void)
{
  int r = f(5);
  __ESBMC_assert(g == 1, "unreachable: the contract is rejected");
  return 0;
}
