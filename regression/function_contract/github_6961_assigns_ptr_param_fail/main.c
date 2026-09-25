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
  b[1] = 7;
  clr(b);
  __ESBMC_assert(b[1] == 7, "b[1] survived the replaced call");
  return 0;
}
