#include <cassert>

extern "C" int nondet_int();

// The member initialiser reaches the adjuster as `m = T(x, y)`, whose call is
// missing the object argument: converting its arguments before the fold
// inserts `&m` matches each against the wrong parameter, and the double is
// round-tripped through int, so 2.5 arrives as 2.0. The negative half
// asserts the value is *not* 2.5: it must fail, and does not if the double is
// truncated.
struct T
{
  int a;
  double b;
  T(int x, double y) : a(x), b(y)
  {
  }
};

struct L
{
  T m;
  L(int x, double y) : m(x, y)
  {
  }
};

int main()
{
  int i = nondet_int();
  L l(i, 2.5);
  assert(l.m.b != 2.5);
}
