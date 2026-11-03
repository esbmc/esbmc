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

struct N
{
  W w;
  int k;
};

struct V
{
  W w;
  M a[2];
  N n;
  V() : w{M(1)}, a{M(2), M(3)}, n{{M(4)}, 5}
  {
  }
};

int main()
{
  {
    V v;
    assert(live == 4);
    assert(v.w.m.v == 1);
    assert(v.a[1].v == 3);
    assert(v.n.w.m.v == 4);
  }
  assert(live == 0);
  return 0;
}
