// R100: a temporary a class-type return value reads only through a member,
// `C(8)` in `H{C(7), C(8).v}`, is destroyed before the function returns.
#include <cassert>

int dtors = 0;

struct C
{
  int v;
  C(int x) : v(x) {}
  ~C() { dtors++; }
};

struct H
{
  C c;
  int w;
};

H wrap() { return H{C(7), C(8).v}; }

int main()
{
  H h = wrap();
  assert(dtors == 0);
}
