#include <cassert>

int dtors = 0;

struct C
{
  int v;
  C(int x) : v(x) {}
  ~C() { dtors++; }
};

C make() { return C(C(4).v + 1); }

int main()
{
  C c = make();
  assert(c.v == 5);
  assert(dtors == 0);
}
