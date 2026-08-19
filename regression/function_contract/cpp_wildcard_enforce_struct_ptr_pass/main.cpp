// C++ variant: --enforce-contract '*' --function sum_point with a pointer-to-
// struct parameter, whose extent the contract has to state like any other.
// const Point * also pins that is_fresh accepts a pointer-to-const argument.

struct Point
{
  int x;
  int y;
};

int sum_point(const Point *p)
{
  __ESBMC_requires(p != nullptr);
  __ESBMC_requires(__ESBMC_is_fresh(p, sizeof(Point)));
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == p->x + p->y);
  return p->x + p->y;
}

int main()
{
  Point pt = {3, 4};
  int res = sum_point(&pt);
  return 0;
}
