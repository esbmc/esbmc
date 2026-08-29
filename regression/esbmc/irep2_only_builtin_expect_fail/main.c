int main(void)
{
  int x = 1;
  __ESBMC_assert(__builtin_expect(x == 2, 1), "expect keeps a false argument false");
  return 0;
}
