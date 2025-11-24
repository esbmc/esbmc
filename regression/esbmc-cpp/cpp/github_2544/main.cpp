#include <cassert>

class A
{
public:
  int x = 0;
  A()
  {
    x = 1;
  }
};

class B : virtual public A
{
public:
  B()
  {
    x = 2;
  }
};

int main()
{
  B b;
  assert(0);
  return 0;
}

