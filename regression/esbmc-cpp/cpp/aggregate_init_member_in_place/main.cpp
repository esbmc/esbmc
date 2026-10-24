#include <cassert>

int ctors = 0, dtors = 0;
const void *built = nullptr;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++ctors;
    built = this;
  }
  M(const M &o) : v(o.v)
  {
    ++ctors;
    built = this;
  }
  ~M()
  {
    ++dtors;
  }
};

struct W
{
  int k;
  M m;
};

struct X
{
  W w;
  M n;
};

int main()
{
  {
    W a{1, M(5)};
    assert(built == &a.m);
    assert(a.k == 1 && a.m.v == 5);
    assert(ctors == 1);
  }
  assert(dtors == 1);

  {
    M t(2);
    W b{3, t};
    assert(built == &b.m);
  }
  assert(ctors == 3 && dtors == 3);

  {
    M arr[2] = {M(6), M(7)};
    assert(built == &arr[1]);
    X x{{4, M(8)}, M(9)};
    assert(built == &x.n);
    assert(arr[0].v == 6 && x.w.m.v == 8 && x.n.v == 9);
  }
  assert(ctors == 7 && dtors == 7);
  return 0;
}
