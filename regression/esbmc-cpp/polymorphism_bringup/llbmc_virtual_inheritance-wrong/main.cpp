/*
 * Multi-inheritance diamond problem
 * non-trivial constructor
 */
#include <cassert>

class A
{
public:
    A() {}
    A(int x) : m_x(x) {};
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
    D(int x) : B(x), C(x) {}
};

int main()
{
    A *a = new D(42);
    assert(a->getX() == 42);
    return 0;
}
