struct P
{
  int x;
  int y;
};

int main(void)
{
  struct P a[3] = {{1, 2}, {3, 4}, {5, 6}};
  a[1].x = 10;
  a[2] = a[0];
  struct P b = a[1];
  __ESBMC_assert(a[1].x == 10 && a[1].y == 4, "element write");
  __ESBMC_assert(a[2].x == 1 && a[2].y == 2, "element copy");
  __ESBMC_assert(b.x == 10 && b.y == 4, "element read");
  return 0;
}
