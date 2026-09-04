struct inner
{
  int n;
  int m;
};

struct outer
{
  struct inner in;
  int k;
};

int main(void)
{
  struct outer x;
  x.in.n = 4;
  x.in.m = 7;
  x.k = 9;

  int s = 0;
  for (int i = 0; i < x.in.n; i++)
    s++;

  __ESBMC_assert(s == 5, "the loop runs one time too many");
  return 0;
}
