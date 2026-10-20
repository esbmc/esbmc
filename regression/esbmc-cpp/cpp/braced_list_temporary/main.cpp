#include <cassert>

int live = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++live;
  }
  M(const M &o) : v(o.v)
  {
    ++live;
  }
  ~M()
  {
    --live;
  }
};

struct W
{
  M m;
};

struct A
{
  M a[2];
};

int value(const W &w)
{
  return w.m.v + live;
}

int main()
{
  {
    W w{M(1)};
    w = W{M(5)};
    assert(live == 1);
    assert(w.m.v == 5);

    A x{M(0), M(0)};
    x = A{{M(2), M(3)}};
    assert(live == 3);
    assert(x.a[1].v == 3);

    int r = value(W{M(6)});
    assert(r == 10);
    assert(live == 3);
  }
  assert(live == 0);
  return 0;
}
