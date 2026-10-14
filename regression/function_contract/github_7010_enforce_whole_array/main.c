// The enforce side read the same clause as `a[0]` and so asserted the whole
// array unchanged, reporting a frame violation against a body that writes only
// what its own clause names.
int a[4];

void f(void)
{
  __ESBMC_assigns(a);
  __ESBMC_ensures(a[1] == 1);
  a[1] = 1;
}

int main(void)
{
  return 0;
}
