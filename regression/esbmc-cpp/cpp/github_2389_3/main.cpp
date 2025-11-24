#include <cassert>

class A
{
public:
  void test()
  {
    assert(0);
  }
};

class B : public A
{
};

int main()
{
}