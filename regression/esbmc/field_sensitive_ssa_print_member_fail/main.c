union u
{
  int i;
  char c[4];
};

struct holder
{
  union u un;
  int t;
};

int main(void)
{
  struct holder v;
  v.un.i = 9;
  v.t = 4;
  __ESBMC_assert(v.t != 4, "t is 4");
  return 0;
}
