// Negative counterpart: the destructor runs exactly once for a local object,
// so g is 1 (not 2, the old double-destruct count). Asserting g == 2 fails.
#include <cassert>

int g;

struct B
{
  int x;
  B() : x(0)
  {
  }
  ~B()
  {
    g++;
  }
};

int main()
{
  {
    B b;
    (void)b;
  }
  assert(g == 2);
  return 0;
}
