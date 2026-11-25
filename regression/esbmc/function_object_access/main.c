int f(void) { return 7; }
int main()
{
  int (*p)(void) = f;
  void *v = (void *)f;
  int x = 0;
  __ESBMC_assert(p() == 7, "call through pointer");
  __ESBMC_assert(v == (void *)&f, "address compare");
  *(int *)&x = 1;
  __ESBMC_assert(x == 1, "data write");
  return 0;
}
