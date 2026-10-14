#include <sstream>
#include <ostream>
#include <cstring>
#include <cassert>

struct Point
{
  int x, y;
};

std::ostream &operator<<(std::ostream &o, const Point &p)
{
  o << p.x << "," << p.y;
  return o;
}

int main()
{
  std::ostringstream ss;
  Point p = {3, 4};
  ss << p;
  assert(strcmp(ss.str().c_str(), "3,4") == 0);
  return 0;
}
