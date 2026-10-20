// A default-initialised array whose constructor is not constexpr is built on
// the first call, one element at a time ([stmt.dcl]/3, [dcl.init]/7).
#include <cassert>
int g = 0;
struct T { int v; T() : v(++g) {} };
T *n() { static T a[2]; return a; }
T (*m())[2] { static T b[2][2]; return b; }
int main() {
  assert(g == 0);
  T *p = n();
  assert(g == 2 && p[0].v == 1 && p[1].v == 2);
  n();
  assert(g == 2);
  T (*q)[2] = m();
  assert(g == 6 && q[0][0].v == 3 && q[1][1].v == 6);
  m();
  assert(g == 6);
  return 0;
}
