int nondet_int(void);

int user_ESBMC_Main_fake(int n)
{
  if (n == 0)
    return 0;
  return user_ESBMC_Main_fake(n - 1);
}

int main(void)
{
  int depth = nondet_int();
  __ESBMC_assume(depth >= 1 && depth <= 3);
  __ESBMC_assert(user_ESBMC_Main_fake(depth) != 0, "recursive result");
}
