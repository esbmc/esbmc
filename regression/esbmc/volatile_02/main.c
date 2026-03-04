int main()
{
  volatile int flag = 1;

  int read = flag;

  __ESBMC_assert(read, "");
}
