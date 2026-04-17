// Nested function passed as function pointer
void call_it(void (*f)(void))
{
  f();
}

int main()
{
  int x = 0;
  void set()
  {
    x = 42;
  }
  call_it(set);
  __ESBMC_assert(x == 42, "nested func called via pointer");
  return 0;
}
