// esbmc/esbmc#938: a virtual base subobject is initialised only by the
// most-derived (complete-object) constructor. Here D lists A(x) in its
// mem-initialiser list, so the shared A::m_x is initialised and the
// assertion holds. Compare llbmc_virtual_inheritance-wrong, where D omits
// A(x) and A::m_x is left indeterminate.
#include <cassert>

class A
{
public:
  A() {}
  A(int x) : m_x(x) {}
  int getX() { return m_x; }

protected:
  int m_x;
};

class B : virtual public A
{
public:
  B() {}
  B(int x) : A(x) {}
};

class C : virtual public A
{
public:
  C() {}
  C(int x) : A(x) {}
};

class D : public B, public C
{
public:
  D() {}
  D(int x) : A(x), B(x), C(x) {}
};

int main()
{
  A *a = new D(42);
  assert(a->getX() == 42);
  return 0;
}
