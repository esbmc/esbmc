int myabs(int a)
{
  if (a < 0)
    return -a;
  return a;
}

int main()
{
  int x;
  __ESBMC_assert(
    __ESBMC_forall(&x, x == (-0x7fffffff - 1) || myabs(x) >= 0),
    "abs nonnegative away from INT_MIN");
}
