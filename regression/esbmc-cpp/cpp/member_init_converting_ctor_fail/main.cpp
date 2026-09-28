// immer::map's `impl_t impl_ = impl_t::empty();`: a converting constructor
// builds the member itself, so no second object is destroyed.
#include <cassert>
int dtors = 0;
struct C
{
  int *root;
  C(int *r, int sz = 0) : root(r)
  {
  }
  ~C()
  {
    dtors++;
  }
  static int *empty()
  {
    static int n;
    return &n;
  }
};
struct M
{
  C impl_ = C::empty();
};
int main()
{
  {
    M m;
    assert(m.impl_.root == C::empty());
  }
  assert(dtors == 2);
  return 0;
}
