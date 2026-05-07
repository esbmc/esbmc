// Reference-form dynamic_cast on a matching type must succeed (no throw),
// preserving the field value through the cast. Companion to dynamic_cast3_bug
// which exercises the bad_cast throw direction.
#include <cassert>
#include <typeinfo>

class Base
{
public:
  virtual ~Base()
  {
  }
};

class Derived : public Base
{
public:
  int x;
};

int main()
{
  Derived d;
  d.x = 42;
  Base &b = d;
  Derived &rd = dynamic_cast<Derived &>(b);
  assert(rd.x == 42);
  return 0;
}
