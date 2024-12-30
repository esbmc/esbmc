#include <cassert>
class A
{
public:
  A(int a) : a(a)
  {
  }
  int a;
};
class B : public A
{
  using A::A;

public:
  int b = 245;
};
class C : public A
{
public:
  C(int a) : A(a)
  {
  }
};

int main()
{
  A a = A(1);
  assert(a.a == 1);
  B b = B(22);
  assert(b.b == 245);
  assert(b.a == 22);
  C c = C(33);
  assert(c.a == 33);
}