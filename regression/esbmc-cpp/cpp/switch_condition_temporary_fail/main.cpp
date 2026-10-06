// The temporary of the switch condition is destroyed before the case label is
// reached ([class.temporary]/4), so one has died by the assignment.
#include <cassert>

int dtors = 0;

struct T
{
  int v;
  T(int x) : v(x)
  {
  }
  ~T()
  {
    ++dtors;
  }
};

int main()
{
  int k = -1;
  switch (T(1).v)
  {
  case 1:
    k = dtors;
    break;
  }
  assert(k == 0);
}
