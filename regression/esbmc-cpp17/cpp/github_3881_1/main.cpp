// Test: converting move constructor from unique_ptr<Derived> to unique_ptr<Base>
#include <cassert>
#include <memory>

class Base
{
public:
  virtual ~Base() = default;
  virtual int value() const = 0;
};

class Derived : public Base
{
public:
  explicit Derived(int v) : v_(v) {}
  int value() const override { return v_; }

private:
  int v_;
};

// Returns unique_ptr<Base> from make_unique<Derived> — requires converting constructor
std::unique_ptr<Base> make_derived(int v)
{
  return std::make_unique<Derived>(v);
}

int main()
{
  std::unique_ptr<Base> p = make_derived(42);
  assert(p != nullptr);
  assert(p->value() == 42);
  return 0;
}
