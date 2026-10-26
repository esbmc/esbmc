int main()
{
  int x = 0;
  while (1)
  {
  ERROR:
    x = 1;
    break;
  }
  __ESBMC_assert(x == 1, "the loop sets x before it breaks");
  return 0;
}
