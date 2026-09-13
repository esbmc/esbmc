// A multi-dimensional array must destroy every leaf element: scheduling only
// one level would reach none of them, since the outer array's element type is
// itself an array rather than a class.
#include <cassert>

int c = 0, d = 0;

struct M
{
  M()
  {
    ++c;
  }
  ~M()
  {
    ++d;
  }
};

int main()
{
  {
    M a[2][3];
  }
  assert(c == 6 && d == 6);
  return 0;
}
