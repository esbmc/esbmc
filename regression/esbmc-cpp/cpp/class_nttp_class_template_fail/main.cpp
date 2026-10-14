// A C++20 class-type template argument: clang cannot generate a USR for the
// specialisation or for the template parameter object it names.
#include <cassert>
struct P { int x; };
template <P p> struct W {
  int v;
  W() : v(p.x) {}
  int get() const { return v + p.x; }
};
int main() { W<P{2}> a; W<P{3}> b; assert(a.get() == 99); assert(b.get() == 6); return 0; }
