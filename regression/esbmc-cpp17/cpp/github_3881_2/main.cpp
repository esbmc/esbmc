// Test: covariant clone() pattern — override returns unique_ptr<Base>
// but implementation returns make_unique<Derived>
#include <cassert>
#include <memory>

class Coord
{
public:
  virtual ~Coord() = default;
  virtual std::unique_ptr<Coord> clone() const = 0;
  virtual int x() const = 0;
};

class IntCoord : public Coord
{
public:
  IntCoord(int x, int y) : x_(x), y_(y) {}

  std::unique_ptr<Coord> clone() const override
  {
    return std::make_unique<IntCoord>(*this);
  }

  int x() const override { return x_; }

private:
  int x_, y_;
};

int main()
{
  std::unique_ptr<Coord> c0 = std::make_unique<IntCoord>(17, 17);
  std::unique_ptr<Coord> c1 = c0->clone();

  assert(c0 != nullptr);
  assert(c1 != nullptr);
  assert(c0->x() == 17);
  assert(c1->x() == 17);
  return 0;
}
