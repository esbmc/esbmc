#include <iostream>

class Base
{
public:
  virtual void display()
  {
    std::cout << "Base class" << std::endl;
  }
};

class Derived : public Base
{
public:
  void display() override
  {
    std::cout << "Derived class" << std::endl;
  }
};

void func(Base *b)
{
  // Attempt to cast Base* to Derived* incorrectly
  // This will fail if b is not actually pointing to a Derived object
  Derived *d = static_cast<Derived *>(b);
  d->display();
}

int main()
{
  Base *basePtr = new Base();
  func(basePtr); // This will lead to undefined behavior
  delete basePtr;
  return 0;
}
