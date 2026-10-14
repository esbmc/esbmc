// A local object with a non-trivial constructor was materialised through a
// temporary and copied into the variable, leaving the temporary with its own
// scope-exit destructor -- so the destructor ran twice. It now constructs in
// place, so `~B` runs exactly once and g is 1 at the end of the scope.
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
  assert(g == 1);
  return 0;
}
