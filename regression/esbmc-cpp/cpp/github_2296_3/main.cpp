#include <cassert>
template <class T>
struct A
{
  int A;
  void foo(T t);
};
template <class d>
struct g
{
  A<d> f;
  int g;
};

struct i
{
  g<i> h;
  int i;
};

int main()
{
  i obj;
  obj.h.f.A = 1;
  obj.h.f.foo(obj);
  assert(obj.h.f.A == 1);
  obj.h.g = 2;
  assert(obj.h.g == 2);
  obj.i = 3;
  assert(obj.i == 3);
  return 0;
}
