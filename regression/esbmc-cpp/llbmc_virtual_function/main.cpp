#include <cassert>

class Base
{
public:
    virtual int f(void) { return 21; }
};

class Derived: public Base
{
public:
    virtual int f(void) { return 42; }
    int x;
};

int main()
{
    Base *o = new Derived();
    int r = o->f();
    delete o;
    assert(r == 42);
    return r;
}
