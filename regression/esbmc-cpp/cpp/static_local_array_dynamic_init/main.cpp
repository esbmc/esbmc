// A function-local static array with a dynamic initializer is initialized
// when control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
int g = 0;
int bump() { return ++g; }
int *n() { static int a[2][2] = {{bump(), 7}, {0, bump()}}; return a[0]; }
int main() {
  assert(g == 0);
  int *p = n();
  assert(g == 2 && p[0] == 1 && p[1] == 7 && p[3] == 2);
  n();
  assert(g == 2);
  return 0;
}
