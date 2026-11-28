struct P
{
  int x;
  int y;
};

int main(void)
{
  struct P a[3] = {{1, 2}, {3, 4}, {5, 6}};
  unsigned i;
  __ESBMC_assume(i < 3);
  a[i].x = 0;
  __ESBMC_assert(a[0].x == 0 || a[1].x == 0 || a[2].x == 0, "write");
  return 0;
}
