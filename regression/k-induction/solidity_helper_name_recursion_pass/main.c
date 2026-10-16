int user_ESBMC_Main_fake(int n)
{
  if (n == 0)
    return 0;
  return user_ESBMC_Main_fake(n - 1);
}

int main(void)
{
  __ESBMC_assert(user_ESBMC_Main_fake(3) == 0, "recursive result");
}
