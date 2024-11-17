#include <stdint.h>
#include <cassert>

class BaseClass
{
public:
  virtual BaseClass *foo(bool var = false) = 0;
};

class DerivedClass : public BaseClass
{
public:
  DerivedClass *foo(bool var = false) override
  {
    return this;
  }
};

int main()
{
  DerivedClass obj;

  assert(obj.foo() == &obj);
  assert(obj.foo(true) == &obj);

  return 0;
}
