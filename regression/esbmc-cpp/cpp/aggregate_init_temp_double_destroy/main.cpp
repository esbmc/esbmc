// Aggregate-initialising a class whose member type has a destructor runs as
// many destructors as constructors: the temporary that initialises the member
// is the member ([dcl.init.aggr]/4), destroyed with the object. ESBMC used to
// destroy the temporary as well, one destructor too many.
#include <cassert>

int ctors = 0, dtors = 0;

struct M
{
  int v;
  M(int x) : v(x)
  {
    ++ctors;
  }
  M(const M &o) : v(o.v)
  {
    ++ctors;
  }
  ~M()
  {
    ++dtors;
  }
};

struct W
{
  M m;
};

int main()
{
  {
    W a{M(5)};
  }
  assert(ctors == dtors);
  return 0;
}
