int helper(int a)
{
  int t = a * 2;
  return t + 1;
}

int main()
{
  int x;
  __ESBMC_assert(__ESBMC_forall(&x, helper(x) == 2 * x), "always odd");
}
