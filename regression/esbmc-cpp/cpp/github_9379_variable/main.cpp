// #9379: converting a member-function pointer held in a variable, rather than
// &B::get itself, still calls get on the start of c instead of its B subobject.
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
  int (B::*q)() = &B::get;
  int (C::*pm)() = q;
  assert((c.*pm)() != 2);
  return 0;
}
