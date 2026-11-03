// Value-initialising an array of a polymorphic class stores each whole
// element; the store must not be taken as one to the element's base subobject.
#include <cassert>
struct B
{
  virtual int f() { return 0; }
};
struct D : B
{
  int v = 7;
  int f() override { return v; }
};
int main()
{
  D *p = new D[2]();
  B *b = &p[1];
  assert(b->f() == 7 && p[0].v == 7);
  delete[] p;
  return 0;
}
