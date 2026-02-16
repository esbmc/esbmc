// Test __ESBMC_old() with function parameters (local state)

void increment_param(int *ptr, int delta)
{
  __ESBMC_requires(ptr != 0 && delta > 0);
  __ESBMC_ensures(*ptr == __ESBMC_old(*ptr) + delta);

  *ptr += delta;
}

int main()
{
  int value = 100;
  increment_param(&value, 25);
  assert(value == 125);
  return 0;
}
