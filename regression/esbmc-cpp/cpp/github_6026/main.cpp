#include <tuple>
#include <utility>

struct Point
{
  int x, y;
  Point(int x_, int y_) : x(x_), y(y_)
  {
  }
};

int add3(int a, int b, int c)
{
  return a + b + c;
}

int main()
{
  // apply with a plain function
  auto t = std::make_tuple(1, 2, 3);
  int r1 = std::apply(add3, t);
  __ESBMC_assert(r1 == 6, "apply with function");

  // apply with a fold-expression lambda
  int r2 = std::apply([](auto... xs) { return (xs + ...); }, t);
  __ESBMC_assert(r2 == 6, "apply with fold-expression lambda");

  // apply with a capturing lambda
  int base = 10;
  auto t2 = std::make_tuple(5);
  int r3 = std::apply([base](int x) { return base + x; }, t2);
  __ESBMC_assert(r3 == 15, "apply with capturing lambda");

  // apply on an empty tuple
  auto t3 = std::make_tuple();
  int r4 = std::apply([]() { return 42; }, t3);
  __ESBMC_assert(r4 == 42, "apply on empty tuple");

  // make_from_tuple constructing an object
  auto t4 = std::make_tuple(3, 4);
  Point p = std::make_from_tuple<Point>(t4);
  __ESBMC_assert(p.x == 3 && p.y == 4, "make_from_tuple");

  // apply on a prvalue tuple (exercises the Tuple&& forwarding path
  // directly, rather than the Tuple& collapse from a named lvalue above)
  int r5 = std::apply(add3, std::make_tuple(4, 5, 6));
  __ESBMC_assert(r5 == 15, "apply on prvalue tuple");

  return 0;
}
