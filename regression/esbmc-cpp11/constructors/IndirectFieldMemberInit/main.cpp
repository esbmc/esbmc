#include <cassert>

class Example
{
public:
  union
  {
    int a;
    float b;
  };

  Example() : a(10)
  {
  }

  int ret()
  {
    return a;
  }
};

int main()
{
  Example ex;
  assert(ex.ret() == 10);
  return 0;
}