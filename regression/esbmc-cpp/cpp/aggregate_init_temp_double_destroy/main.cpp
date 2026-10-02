// Aggregate-initialising a class from a temporary of a member type with a
// destructor constructs the member in place: one constructor, one destructor.
// ESBMC built the temporary, copied it into the member bitwise and destroyed
// both.
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
