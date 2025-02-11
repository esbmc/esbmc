#include <iostream>

class Foo {
  public:
    Foo() {value = 0;};
    void Execute();
    void Increment();
  private:
    int value;
};

void Foo::Execute() {
        printf("Executing...");
}

void Foo::Increment() {
    value++;
}

int main()
{
  Foo *foo = new Foo();

  delete foo;

  foo->Increment();  // Incrementing the value

  foo->Execute(); // Executing

  return 0;
}
