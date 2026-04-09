// Test __ESBMC_old() with complex expression
// NOTE: __ESBMC_old() requires an lvalue. Instead of __ESBMC_old(x + y),
// snapshot each variable separately.

int x = 0;
int y = 0;

void compute(int a, int b)
{
  __ESBMC_requires(a > 0 && b > 0);
  __ESBMC_ensures(x == __ESBMC_old(x) + a);
  __ESBMC_ensures(y == __ESBMC_old(y) + b);

  x += a;
  y += b;
}

int main()
{
  x = 10;
  y = 20;
  compute(5, 3);
  assert(x == 15);
  assert(y == 23);
  assert(x + y == 38);
  return 0;
}
