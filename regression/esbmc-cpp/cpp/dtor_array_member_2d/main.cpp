// Multidimensional array member: build_destructor_chain recurses through
// each array dimension so every leaf element is destroyed. Here ~C runs
// ~B on all 2*3 elements of the B a[2][3] member (g == 6).
#include <cassert>

int g;

struct B
{
  int x;
  ~B()
  {
    g++;
  }
};

struct C
{
  B a[2][3];
};

int main()
{
  C *c = new C();
  delete c;
  assert(g == 6);
  return 0;
}
