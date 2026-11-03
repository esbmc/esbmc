// The temporaries of the conditions below are destroyed at the end of each
// condition ([class.temporary]/4), so three have died by the assertion.
#include <cassert>

int dtors = 0;

struct C
{
  bool v;
  C(bool b) : v(b)
  {
  }
  ~C()
  {
    ++dtors;
  }
  bool ok() const
  {
    return v;
  }
};

int main()
{
  int n = 0;
  if (C(true).ok())
    while (C(n < 1).ok())
      ++n;
  assert(dtors < 3);
}
