// #9379: &B::get converted to int (C::*)() is called on the B subobject of c
// ([conv.mem]/2), so it returns c.b.
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
  int get()
  {
    return b;
  }
};
struct C : A, B
{
};
int main()
{
  C c;
  int (C::*pm)() = &B::get;
  assert((c.*pm)() != 2);
  return 0;
}
