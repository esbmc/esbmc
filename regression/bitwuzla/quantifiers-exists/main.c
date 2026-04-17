int main()
{
  int value;
  __ESBMC_assert(__ESBMC_exists(&value, value == 42), "exists 42");
  __ESBMC_assert(
    !__ESBMC_exists(&value, value == 0 && value == 1), "contradiction");
  return 0;
}
