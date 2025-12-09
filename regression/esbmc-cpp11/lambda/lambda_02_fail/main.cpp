#include <cassert>

class classA
{
public:
  template <typename T>
  static bool bar(T var)
  {
    var();
    return true;
  }
};

class classB
{
public:
  bool foo()
  {
    return classA::bar([this]() { data++; });
  }
  int data = 0;
};

int main()
{
  classB obj;
  obj.foo();
  assert(obj.data == 0);

  return 0;
}
