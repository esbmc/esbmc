#include <any>
#include <cassert>

struct Point
{
  int x;
  int y;
};

int main()
{
  // Primitive value.
  std::any a = 7;
  assert(a.has_value());
  assert(std::any_cast<int>(a) == 7);

  // Reassign with a different stored type.
  a = 3.14;
  assert(a.has_value());
  assert(std::any_cast<double>(a) == 3.14);

  // Struct value.
  std::any b = Point{1, 2};
  Point p = std::any_cast<Point>(b);
  assert(p.x == 1);
  assert(p.y == 2);

  // reset() clears.
  b.reset();
  assert(!b.has_value());

  return 0;
}
