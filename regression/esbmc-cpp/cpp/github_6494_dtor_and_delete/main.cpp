// github #6494: a class destructor and a replaced operator delete travel in
// the same sideeffect2t::arguments vector (slot 0 and slot 1 respectively),
// so this pins that neither displaces the other across the irep2 round trip.
#include <cstddef>
#include <cassert>
static int deletes = 0;
static int dtors = 0;
void operator delete(void *) noexcept { deletes++; }
void operator delete(void *, size_t) noexcept { deletes++; }
struct C { int v; C() : v(3) {} ~C() { dtors++; } };
int main() {
  C *p = new C();
  assert(p->v == 3);
  delete p;
  assert(dtors == 1);    // destructor must still run (arguments[0])
  assert(deletes == 1);  // replaced delete must run too (arguments[1])
  return 0;
}
