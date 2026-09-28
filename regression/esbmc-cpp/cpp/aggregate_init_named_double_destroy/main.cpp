// KNOWNBUG, the named-argument half of aggregate_init_temp_double_destroy: the
// imbalance is not caused by a temporary in the initialiser. Here the argument
// is an ordinary named object, and ESBMC still runs 2 constructors and 3
// destructors where g++ runs 2 and 2, because the member is copied through a
// helper that is destroyed in addition to the member.
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
