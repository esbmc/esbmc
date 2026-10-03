#include <cassert>

int dtors = 0;

struct C
{
  int v;
  C(int x) : v(x) {}
  ~C() { dtors++; }
};

int member() { return C(1).v; }

int main()
{
  member();
  assert(dtors == 0);
}
