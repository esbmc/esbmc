// Test: converting move assignment operator from unique_ptr<Derived> to unique_ptr<Base>
#include <cassert>
#include <memory>

class Animal
{
public:
  virtual ~Animal() = default;
  virtual int sound() const = 0;
};

class Dog : public Animal
{
public:
  int sound() const override { return 1; }
};

class Cat : public Animal
{
public:
  int sound() const override { return 2; }
};

int main()
{
  // Test converting move assignment
  std::unique_ptr<Animal> a;
  a = std::make_unique<Dog>();
  assert(a != nullptr);
  assert(a->sound() == 1);

  a = std::make_unique<Cat>();
  assert(a != nullptr);
  assert(a->sound() == 2);

  return 0;
}
