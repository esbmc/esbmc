// Test multiple __ESBMC_old() in same ensures clause

int x = 0;
int y = 0;

void swap(void)
{
  __ESBMC_ensures(x == __ESBMC_old(y));
  __ESBMC_ensures(y == __ESBMC_old(x));

  int temp = x;
  x = y;
  y = temp;
}

int main()
{
  x = 5;
  y = 10;
  swap();
  assert(x == 10);
  assert(y == 5);
  return 0;
}
