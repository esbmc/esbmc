#include <cassert>

class Foo {
  public:
    Foo() {};
    ~Foo() {};
    void Execute() { assert(1); }

};

int main()
{
  Foo foo = Foo();
  foo.Execute();
  return 0;
}
