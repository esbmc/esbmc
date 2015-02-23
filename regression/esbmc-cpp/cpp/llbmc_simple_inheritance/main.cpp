#include <cassert>

class Base
{
public:
    int f(void) { return 21; }
};

class Derived: public Base
{
public:
    int f(void) { return 42; }
    int x;
};

int main()
{
    Base *o = new Derived();
    int r = o->f();
    delete o;
    assert(r == 21);
    return r;
}
