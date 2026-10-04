// A temporary bound to a reference member of a braced aggregate initialiser
// lives as long as the aggregate ([class.temporary]/6); other temporaries in
// the initialiser die at the end of the full-expression.
#include <cassert>

int live = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++live;
  }
  ~M()
  {
    --live;
  }
};

int get(const M &m)
{
  return m.v;
}

struct R
{
  const M &m;
};

struct P
{
  const M &m;
  int k;
};

struct O
{
  R r;
  int k;
};

struct RR
{
  M &&m;
};

int main()
{
  {
    R x{M(1)};
    assert(live == 1 && x.m.v == 1);
  }
  assert(live == 0);
  {
    P p = {M(1), get(M(2))};
    assert(live == 1 && p.k == 2);
  }
  assert(live == 0);
  {
    O o{{M(1)}, 2};
    R a[2] = {{M(2)}, {M(3)}};
    RR r{M(4)};
    assert(live == 4 && a[1].m.v == 3 && r.m.v == 4);
  }
  assert(live == 0);
  for (int i = 0; i < 2; ++i)
  {
    R x{M(i)};
    assert(live == 1);
  }
  assert(live == 0);
  return 0;
}
