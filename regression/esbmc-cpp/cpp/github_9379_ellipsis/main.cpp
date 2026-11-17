// #9379: a variadic member function cannot be forwarded, so &B::get converted
// to a pointer to member of C still calls get on the start of c.
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
  int get(int, ...)
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
  int (C::*pm)(int, ...) = &B::get;
  assert((c.*pm)(0) != 2);
  return 0;
}
