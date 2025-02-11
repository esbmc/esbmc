#include <cassert>

template <class T>
class A
{
public:
  T foo(T t)
  {
    return t;
  }
};

template <class d>
class g
{
public:
  A<d> f;
};

class i
{
public:
  int a = 1;
  g<i> h;
};

int main()
{
  i obj;
  assert(obj.h.f.foo(obj).a != 1);
}
