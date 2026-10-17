#include <cassert>

int ctors = 0, dtors = 0;

struct C
{
  int v;
  C(int x) : v(x) { ctors++; }
  C(const C &o) : v(o.v) { ctors++; }
  ~C() { dtors++; }
};

struct H
{
  C c;
  int k;
};

C make() { return C(C(4).v + 1); }

H wrap() { return H{C(C(2).v), 3}; }

C pick(bool b) { return C(b ? C(1).v : 6); }

int main()
{
  {
    C c = make();
    assert(c.v == 5);
    assert(ctors == 2);
    assert(dtors == 1);
  }
  assert(dtors == 2);
  {
    H h = wrap();
    assert(h.c.v == 2);
    assert(ctors == 4);
    assert(dtors == 3);
  }
  assert(dtors == 4);
  {
    C p = pick(false);
    assert(p.v == 6);
  }
  assert(ctors == 5);
  assert(dtors == 5);
}
