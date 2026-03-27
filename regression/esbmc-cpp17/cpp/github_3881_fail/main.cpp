// Test: assertion fails when value is incorrect after clone
#include <cassert>
#include <memory>

class Shape
{
public:
  virtual ~Shape() = default;
  virtual std::unique_ptr<Shape> clone() const = 0;
  virtual int id() const = 0;
};

class Circle : public Shape
{
public:
  explicit Circle(int id) : id_(id) {}

  std::unique_ptr<Shape> clone() const override
  {
    return std::make_unique<Circle>(*this);
  }

  int id() const override { return id_; }

private:
  int id_;
};

int main()
{
  std::unique_ptr<Shape> s = std::make_unique<Circle>(7);
  std::unique_ptr<Shape> s2 = s->clone();
  // This assertion should fail: id is 7 not 99
  assert(s2->id() == 99);
  return 0;
}
