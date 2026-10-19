// #8679: a call through a pointer to a virtual member function dispatches on
// the object's dynamic type.
#include <cassert>

struct B
{
  virtual int g()
  {
    return 1;
  }
  virtual void set(int v)
  {
    x = v;
  }
  int x = 0;
};

struct D : B
{
  int g() override
  {
    return 2;
  }
  void set(int v) override
  {
    x = v + 10;
  }
};

int main()
{
  D d;
  B b;
  B *pb = &d;
  int (B::*pg)() = &B::g;
  void (B::*ps)(int) = &B::set;
  assert((pb->*pg)() == 2);
  assert((b.*pg)() == 1);
  (pb->*ps)(5);
  assert(d.x == 15);
  (b.*ps)(5);
  assert(b.x == 5);
  assert(pg == &B::g);
  return 0;
}
