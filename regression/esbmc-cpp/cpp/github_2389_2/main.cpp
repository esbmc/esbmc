#include <cassert>

class A
{
public:
  void test()
  {
    assert(0);
  }
};

class B
{
public:
  void test()
  {
    assert(1);
  }
};

int main()
{
}