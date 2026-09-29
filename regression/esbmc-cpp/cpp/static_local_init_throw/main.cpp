// A function-local static with a dynamic initializer is initialized when
// control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
int tries = 0;
int init() { if (++tries == 1) throw 1; return 42; }
int get() { static int v = init(); return v; }
int main() {
  try { get(); assert(0); } catch (int) {}
  assert(get() == 42 && tries == 2);
  assert(get() == 42 && tries == 2);
  return 0;
}
