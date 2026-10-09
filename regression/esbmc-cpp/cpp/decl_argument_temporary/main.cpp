// A temporary built for a declaration's constructor or by-value call is
// destroyed at the end of that full-expression ([class.temporary]/4), not at
// the end of the block.
#include <cassert>

int live = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
    ++live;
  }
  C(const C &o) : v(o.v)
  {
    ++live;
  }
  ~C()
  {
    --live;
  }
};

struct D
{
  int w;
  D(C c) : w(c.v)
  {
  }
  D(const C &a, const C &b) : w(a.v + b.v)
  {
  }
  ~D()
  {
  }
};

struct E
{
  int w;
  E(int x) : w(x)
  {
  }
};

int get(const C &c)
{
  return c.v;
}

D make(const C &c)
{
  return D(c, c);
}

int main()
{
  D d(C(6));
  assert(live == 0 && d.w == 6);
  D e{C(1), C(2)};
  assert(live == 0 && e.w == 3);
  E g(get(C(5)));
  assert(live == 0 && g.w == 5);
  D h = D(C(7));
  assert(live == 0 && h.w == 7);
  D m = make(C(4));
  assert(live == 0 && m.w == 8);
  return 0;
}
