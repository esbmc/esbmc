// A function-local static with a dynamic initializer is initialized when
// control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
int ctors = 0;
struct T { int v; T(int x) : v(x) { ++ctors; } };
int get(int x) { static T t(x); return t.v; }
int main() {
  assert(ctors == 0);
  assert(get(3) == 3);
  assert(get(9) == 3);
  assert(ctors == 1);
  return 0;
}
