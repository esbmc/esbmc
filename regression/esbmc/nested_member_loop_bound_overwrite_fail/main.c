struct inner
{
  int n;
};

struct outer
{
  struct inner in;
};

int main(void)
{
  struct outer x;
  x.in.n = 4;
  x.in.n = 6;

  int s = 0;
  for (int i = 0; i < x.in.n; i++)
    s++;

  __ESBMC_assert(s == 4, "the loop reads the first write");
  return 0;
}
