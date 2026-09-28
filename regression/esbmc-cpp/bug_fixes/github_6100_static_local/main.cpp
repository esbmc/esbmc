int calls;

int mkinit()
{
  return ++calls;
}

int f()
{
  static int n = mkinit();
  return n;
}

int main()
{
  int x;
  int first = f();
  __ESBMC_assert(__ESBMC_forall(&x, f() == first), "static inited once");
}
