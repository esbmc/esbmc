// A temporary bound to a reference member of a braced aggregate lives as long
// as the aggregate; a parenthesised one dies with its full-expression
// ([class.temporary]/6). Both assertions fail natively.
#include <cassert>
int dtors;
struct M
{
  int v;
  M(int x) : v(x)
  {
  }
  ~M()
  {
    ++dtors;
  }
};
struct R
{
  const M &m;
};
int main()
{
  {
    R x{M(1)};
    assert(dtors == 1);
  }
  {
    R y(M(2));
    assert(dtors == 1);
  }
  return 0;
}
