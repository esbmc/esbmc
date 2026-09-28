// A function-local static with a dynamic initializer is initialized when
// control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
int g = 0;
int bump() { return ++g; }
void n() { static int c = bump(); (void)c; }
int main() {
  assert(g == 1);
  return 0;
}
