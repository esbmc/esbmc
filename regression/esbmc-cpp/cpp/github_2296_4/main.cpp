#include <cassert>
struct b;
struct a
{
  int a;
  void foo(b);
};
struct b
{
  int b;
  void foo(a);
};

int main()
{
  a obj1;
  b obj2;
  obj1.a = 1;
  obj2.b = 2;
  obj1.foo(obj2);
  assert(obj1.a == 1);
  assert(obj2.b == 2);
  return 0;
}
