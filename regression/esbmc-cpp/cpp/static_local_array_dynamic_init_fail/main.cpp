// A function-local static array with a dynamic initializer is initialized
// when control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
int g = 0;
int bump() { return ++g; }
void n() { static int a[2] = {bump(), bump()}; (void)a; }
int main() {
  assert(g == 2);
  return 0;
}
