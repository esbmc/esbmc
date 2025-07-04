#include <cassert>

class A
{
public:
  void test()
  {
    assert(1);
  }
};

class B
{
public:
  void test()
  {
    assert(0);
  }
};

int main()
{
}