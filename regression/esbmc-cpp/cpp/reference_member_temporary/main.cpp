// A temporary bound to a reference member of a braced aggregate lives as long
// as the aggregate; a parenthesised one dies with its full-expression
// ([class.temporary]/6).
#include <cassert>
int dtors;
struct M
{
  int *p;
  M(int v) : p(new int(v))
  {
  }
  M(const M &) = delete;
  ~M()
  {
    delete p;
    ++dtors;
  }
};
struct R
{
  const M &m;
  int k;
};
struct O
{
  R r;
  const M &n;
};
int main()
{
  {
    R x{M(1), 0};
    assert(dtors == 0 && *x.m.p == 1);
    O o{{M(2), 0}, M(3)};
    assert(dtors == 0 && *o.r.m.p + *o.n.p == 5);
    R a[2] = {{M(4), 0}, {M(5), 0}};
    assert(dtors == 0 && *a[1].m.p == 5);
  }
  assert(dtors == 5);
  R y(M(6), 0);
  assert(dtors == 6);
  return 0;
}
