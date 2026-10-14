int main(void)
{
  int x = 1;
  __ESBMC_assert(__builtin_expect(x == 1, 0), "expect yields its first argument");
  return 0;
}
