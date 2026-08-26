// esbmc/esbmc#7025: B::put writes buf, never the virtual base V, so the
// assertion below must not hold once `this` lands on the B subobject.
#include <cassert>

struct V
{
  int v;
};
struct A
{
  int a;
};
struct B : virtual V
{
  char buf[8];
  void put(char c)
  {
    buf[0] = c;
  }
};
struct D : A, B
{
};

int main()
{
  D d;
  d.v = 0;
  d.put('A');
  // B::put writes buf, never the virtual base: this must not hold.
  assert(d.v == 'A');
  return 0;
}
