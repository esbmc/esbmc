// Simpler case: class static object used by another static member's initializer
struct Point
{
  static const Point ORIGIN;
  static const int X_DOUBLE;
  static const int Y_DOUBLE;

  int x, y;
  constexpr Point(int x, int y) : x(x), y(y) {}
};

constexpr Point Point::ORIGIN = Point(3, 4);
constexpr int Point::X_DOUBLE = Point::ORIGIN.x * 2;
constexpr int Point::Y_DOUBLE = Point::ORIGIN.y * 2;

int main()
{
  __ESBMC_assert(Point::ORIGIN.x == 3, "origin-x");
  __ESBMC_assert(Point::ORIGIN.y == 4, "origin-y");
  __ESBMC_assert(Point::X_DOUBLE == 6, "x-double");
  __ESBMC_assert(Point::Y_DOUBLE == 8, "y-double");
  return 0;
}
