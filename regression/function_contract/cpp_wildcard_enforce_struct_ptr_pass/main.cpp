// C++ variant: --enforce-contract '*' --function sum_point with a pointer-to-
// struct parameter.  Exercises the struct/union branch of
// add_pointer_validity_assumptions, which allocates a single stack element
// (not a malloc array) so that struct-field SSA phi-nodes are handled correctly.

struct Point
{
  int x;
  int y;
};

int sum_point(const Point *p)
{
  __ESBMC_requires(p != nullptr);
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
