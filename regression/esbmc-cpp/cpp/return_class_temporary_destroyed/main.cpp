// R79's residual: a temporary in a class-type return value that is not the
// returned object is destroyed before the function returns.
#include <cassert>

int dtors = 0;

struct C
{
  int v;
  C(int x) : v(x) {}
  ~C() { dtors++; }
};

struct H
{
  C c;
  int w;
};

C make() { return C(C(4).v + 1); }
H wrap() { return H{C(7), C(8).v}; }

int main()
{
  {
    C c = make();
    assert(c.v == 5);
    assert(dtors == 1);
  }
  assert(dtors == 2);
  {
    H h = wrap();
    assert(h.c.v == 7 && h.w == 8);
    assert(dtors == 3);
  }
  assert(dtors == 4);
}
