// The elements `{}` leaves to its array filler are constructed by a
// constructor that is not constexpr, so the initialization is dynamic and
// runs on the first call ([basic.start.static]/2, [stmt.dcl]/3).
#include <cassert>
int g = 0;
struct T { int v; T() : v(++g) {} };
T *n() { static T a[2] = {}; return a; }
int main() {
  assert(g == 0);
  T *p = n();
  assert(g == 2 && p[0].v == 1 && p[1].v == 2);
  n();
  assert(g == 2);
  return 0;
}
