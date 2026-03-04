int main()
{
  volatile int flag = 0;

  while (flag != 0)
  {
    __ESBMC_assert(0, "");
  }
}
