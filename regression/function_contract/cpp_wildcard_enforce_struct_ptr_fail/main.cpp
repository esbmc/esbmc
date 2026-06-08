// Companion fail case: ensures claims p->x + p->y + 1 but body returns
// p->x + p->y.

struct Point
{
  int x;
  int y;
};

int sum_point(const Point *p)
{
  __ESBMC_requires(p != nullptr);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == p->x + p->y + 1); // wrong
  return p->x + p->y;
}

int main()
{
  Point pt = {3, 4};
  int res = sum_point(&pt);
  return 0;
}
