// Test __ESBMC_old() functionality with a simple global variable

int global = 0;

void increment_global(int x)
{
  __ESBMC_requires(x > 0);
  __ESBMC_ensures(global == __ESBMC_old(global) + x);

  global += x;
}

int main()
{
  global = 10;
  increment_global(5);
  assert(global == 15);
  return 0;
}
