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

// KNOWNBUG: this invalid down-cast + virtual call is UB, but with the
// ABI-correct nested-base / primary-base-vptr-sharing layout it is NOT detected.
// Base and Derived are both 8 bytes (shared vptr at offset 0, Derived adds no
// members), so there is no spatial memory error to catch. Master's old flat
// vtable layout modeled Derived with two vptrs (16 bytes) and faulted reading
// the second vptr at offset 8 of an 8-byte object — an artifact of the
// non-ABI layout, not a real C++ memory error. Detecting the actual UB (the
// object's dynamic type is not Derived) soundly needs RTTI-style dynamic-type
// checking at every virtual call (reusing build_dynamic_cast's vptr-vs-allowed-
// vtable-address comparison) while handling partially-constructed objects and
// merged/thunk tables — a feature, not a patch. Tracked as a follow-up.
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
