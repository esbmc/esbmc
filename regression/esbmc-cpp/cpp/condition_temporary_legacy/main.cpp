// A temporary in an if, while, for or do-while condition is destroyed at the
// end of the condition, before the branch ([class.temporary]/4).
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
  if (C(true).ok())
    assert(dtors == 1);

  int n = 0;
  while (C(n < 2).ok())
  {
    assert(dtors == 2 + n);
    ++n;
  }
  assert(dtors == 4);

  for (n = 0; C(n < 1).ok(); ++n)
    assert(dtors == 5);
  assert(dtors == 6);

  do
    ++n;
  while (C(false).ok());
  assert(dtors == 7);
}
