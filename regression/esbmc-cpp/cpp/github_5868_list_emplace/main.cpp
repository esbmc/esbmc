#include <list>
#include <cassert>

struct point
{
  int x;
  int y;
  point() : x(0), y(0)
  {
  }
  point(int a, int b) : x(a), y(b)
  {
  }
};

int main()
{
  std::list<point> l;
  point &b = l.emplace_back(1, 2);
  assert(b.x == 1 && b.y == 2);
  assert(l.size() == 1);
  assert(l.back().x == 1);

  point &f = l.emplace_front(3, 4);
  assert(f.x == 3);
  assert(l.size() == 2);
  assert(l.front().y == 4);
  assert(l.back().x == 1);

  std::list<int> n;
  n.emplace_back(5);
  n.emplace(n.begin(), 6);
  assert(n.size() == 2);
  assert(n.front() == 6);
  assert(n.back() == 5);

  // The returned reference aliases the element.
  n.emplace_back(7) = 8;
  assert(n.back() == 8);
  return 0;
}
