#include <cassert>

class classA
{
public:
  template <typename T>
  static bool bar(T var)
  {
    return true;
  }
};

class classB
{
public:
  classB()
  {
  }

  bool foo()
  {
    return classA::bar([this]() { return true; });
  }
};

int main()
{
  classB obj;
  assert(obj.foo() == false);

  return 0;
}
