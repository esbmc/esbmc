// Negative variant of github_4715_irep2_bodies_cpp_01: same constructs but
// with a wrong assertion that ESBMC must falsify.

struct Point
{
  int x, y;
  Point(int a, int b) : x(a), y(b)
  {
  }
  Point operator+(const Point &o) const
  {
    return Point(x + o.x, y + o.y);
  }
};

int main()
{
  Point p(3, 4);
  Point q = p + Point(1, 2);
  // q.x == 4, q.y == 6, q.x + q.y == 10 — asserting 11 must fail.
  __ESBMC_assert(q.x + q.y == 11, "wrong sum");
  return 0;
}
