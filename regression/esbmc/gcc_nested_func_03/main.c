// Nested function with own parameters and capture
int main()
{
  int base = 10;
  int add(int y)
  {
    return base + y;
  }
  int r = add(5);
  __ESBMC_assert(r == 15, "nested func with params and capture");
  return 0;
}
