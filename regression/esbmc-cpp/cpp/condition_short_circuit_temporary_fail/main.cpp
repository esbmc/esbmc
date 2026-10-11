// The temporary in the right operand of && is destroyed at the end of each
// evaluation of the loop condition ([class.temporary]/4), so three have died.
#include <cassert>

int dtors = 0;

struct C
{
  ~C()
  {
    ++dtors;
  }
  bool ok() const
  {
    return true;
  }
};

int main()
{
  int n = 0;
  while (n < 3 && C().ok())
    ++n;
  assert(dtors == 0);
}
