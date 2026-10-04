// R79's residual: a temporary in a class-type return value that is not the
// returned object is destroyed before the function returns.
#include <cassert>

int dtors = 0;

struct C
{
  int v;
  C(int x) : v(x) {}
  ~C() { dtors++; }
};

C make() { return C(C(4).v + 1); }

int main()
{
  C c = make();
  assert(dtors == 0);
}
