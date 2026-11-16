// #9379: a member-function pointer converted from a non-first base binds
// `this` to that base's subobject, also through a pointer and for a virtual
// member of a deeper derived class and in a hierarchy with a virtual base.
#include <cassert>
struct A
{
  int a;
  A() : a(1)
  {
  }
};
struct B
{
  int b;
  B() : b(2)
  {
  }
  void set(int v)
  {
    b = v;
  }
  int add(int x, int y)
  {
    return b + x + y;
  }
  virtual int vget()
  {
    return b;
  }
};
struct C : A, B
{
};
struct D : C
{
  int d;
  D() : d(4)
  {
  }
  int vget()
  {
    return d;
  }
};
struct V
{
  int v;
  V() : v(7)
  {
  }
};
struct E : virtual V, A, B
{
};
int main()
{
  C c;
  void (C::*ps)(int) = &B::set;
  (c.*ps)(5);
  assert(c.b == 5);
  assert(c.a == 1);
  C *cp = &c;
  int (C::*pa)(int, int) = &B::add;
  assert((cp->*pa)(1, 2) == 8);
  D d;
  int (D::*pv)() = &B::vget;
  assert((d.*pv)() == 4);
  E e;
  int (E::*pe)(int, int) = &B::add;
  assert((e.*pe)(0, 1) == 3);
  return 0;
}
