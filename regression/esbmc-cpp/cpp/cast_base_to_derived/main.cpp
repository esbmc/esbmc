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
  Derived *d = static_cast<Derived *>(b);
  d->display();
}

int main()
{
  Base *basePtr = new Derived();
  func(basePtr);
  delete basePtr;
  return 0;
}
