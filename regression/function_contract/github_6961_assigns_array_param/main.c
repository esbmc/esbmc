#define N 4

void clr(int a[N])
{
  __ESBMC_assigns(a);
  __ESBMC_ensures(a[0] == 0);

  for (int i = 0; i < N; i++)
    a[i] = 0;
}

int main(void)
{
  int b[N];
  b[0] = 1;
  clr(b);
  __ESBMC_assert(b[0] == 0, "callee zeroed the first element");
  return 0;
}
