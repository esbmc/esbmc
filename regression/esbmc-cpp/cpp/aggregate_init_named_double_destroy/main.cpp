// The named-argument half of aggregate_init_temp_double_destroy: the member is
// copy-constructed from a named object, and that copy is the member, so g++
// runs 2 constructors and 2 destructors.
#include <cassert>

int c = 0, d = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++c;
  }
  M(const M &o) : v(o.v)
  {
    ++c;
  }
  ~M()
  {
    ++d;
  }
};

struct W
{
  M m;
};

int main()
{
  {
    M t(5);
    W a{t};
    assert(a.m.v == 5); // the stored value is correct
  }
  assert(c == d);
  return 0;
}
