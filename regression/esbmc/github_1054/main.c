int main()
{
  unsigned int a;

  if(a < 10)
  {
    __ESBMC_assert(a > 10, "");
  }

  return 0;
}