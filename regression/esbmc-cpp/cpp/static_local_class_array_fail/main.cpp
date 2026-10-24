// A default-initialised array whose constructor is not constexpr is built on
// the first call, one element at a time ([stmt.dcl]/3, [dcl.init]/7).
#include <cassert>
int g = 0;
struct T { int v; T() : v(++g) {} };
void n() { static T a[2]; (void)a; }
int main() {
  assert(g == 2);
  return 0;
}
