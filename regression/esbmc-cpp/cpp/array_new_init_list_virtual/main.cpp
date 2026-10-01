// Listed elements of a polymorphic class are constructed in place, so each
// dispatches through its own vtable pointer.
#include <cassert>
struct B
{
  virtual int f() { return 0; }
};
struct D : B
{
  int v;
  D(int x) : v(x) {}
  int f() override { return v; }
};
int main()
{
  D *p = new D[2]{D(5), D(6)};
  B *b = &p[1];
  assert(b->f() == 6);
  delete[] p;
  return 0;
}
