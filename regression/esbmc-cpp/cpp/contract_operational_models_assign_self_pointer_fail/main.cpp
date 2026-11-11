// S's copy points `self` at its own member, so a copy is not byte-equal to
// the source and assign's contract must not say it is.
#include <vector>

struct S
{
  int v;
  int *self;
  S() : v(0), self(&v)
  {
  }
  S(const S &o) : v(o.v), self(&v)
  {
  }
  S &operator=(const S &o)
  {
    v = o.v;
    return *this;
  }
};

int main()
{
  S s;
  std::vector<S> vec;
  vec.assign(1, s);
  __ESBMC_assert(vec[0].self == &s.v, "false: the copy points at s");
  return 0;
}
