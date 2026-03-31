int main()
{
  int SIZE = 16;
  int vec[SIZE];
  int x = vec[0];
  __ESBMC_assert(SIZE == 16, "size is 16");
  return 0;
}
