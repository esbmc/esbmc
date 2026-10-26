// A comma's right operand initialises the variable ([expr.comma]/1): a class
// prvalue there is not copied out of a temporary that is then destroyed.
#include <cassert>

int dtors = 0;

struct A
{
  int *p;
  A(int v) : p(new int(v))
  {
  }
  A(const A &) = delete;
  ~A()
  {
    delete p;
    ++dtors;
  }
};

A make(int v)
{
  return A(v);
}

int main()
{
  int k = 0;
  {
    A a = (++k, A(3));
    assert(*a.p == 3);
    assert(dtors == 0);
    A b = (++k, make(4));
    assert(*b.p == 4);
    auto c = (++k, ++k, A(5));
    assert(*c.p == 5);
    assert(dtors == 0);
  }
  assert(k == 4);
  assert(dtors == 3);
  return 0;
}
