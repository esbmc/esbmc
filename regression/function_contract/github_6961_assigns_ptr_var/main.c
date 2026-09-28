#define N 4

void clr(int *p)
{
  __ESBMC_assigns(p);
  __ESBMC_ensures(p[0] == 0);

  for (int i = 0; i < N; i++)
    p[i] = 0;
}

int main(void)
{
  int b[N];
  int *q = b;
  b[0] = 1;
  clr(q);
  __ESBMC_assert(b[0] == 0, "callee zeroed the first element");
  return 0;
}
