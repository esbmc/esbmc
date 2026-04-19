// Test __ESBMC_old() with boolean type

_Bool flag = 0;

void toggle_flag(void)
{
  __ESBMC_ensures(flag == !__ESBMC_old(flag));

  flag = !flag;
}

int main()
{
  flag = 0;
  toggle_flag();
  assert(flag == 1);
  return 0;
}
