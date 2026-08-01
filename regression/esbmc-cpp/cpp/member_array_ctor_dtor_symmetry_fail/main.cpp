// Negative twin: the elements really are constructed, so a destructor that
// asserts the opposite must be reported rather than passing vacuously.
#include <cassert>

int marker;

struct E
{
  int *p;
  E()
  {
    p = &marker;
  }
  ~E()
  {
    assert(p != &marker);
  }
};

struct H
{
  E buf[20];
};

int main()
{
  H h;
  return 0;
}
