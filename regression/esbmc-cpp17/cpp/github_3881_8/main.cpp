// Test: full example from discussion #3881 — assign, clone, and operator==
// across a polymorphic hierarchy using unique_ptr<Base>.
// operator== is dispatched through a pure virtual equals(); comparing
// *coord0 == *coord1 after *coord1 = *coord0 must succeed, and a clone
// of coord0 must also be equal to it.
#include <memory>

class Coord
{
public:
  virtual ~Coord() = default;
  virtual std::unique_ptr<Coord> clone() const = 0;
  virtual void assign(const Coord &other) = 0;
  virtual bool equals(const Coord &other) const = 0;

  Coord &operator=(const Coord &other)
  {
    assign(other);
    return *this;
  }

  bool operator==(const Coord &other) const { return equals(other); }
  virtual void inc() = 0;
};

class IntCoord : public Coord
{
  int x, y;

public:
  IntCoord(int xnum, int ynum) : x(xnum), y(ynum) {}

  std::unique_ptr<Coord> clone() const override
  {
    return std::make_unique<IntCoord>(*this);
  }

  void assign(const Coord &other) override
  {
    const auto &o = dynamic_cast<const IntCoord &>(other);
    x = o.x;
    y = o.y;
  }

  bool equals(const Coord &other) const override
  {
    const auto &o = dynamic_cast<const IntCoord &>(other);
    return x == o.x && y == o.y;
  }

  void inc() override { ++x; ++y; }
};

int main()
{
  std::unique_ptr<Coord> coord0 = std::make_unique<IntCoord>(17, 17);
  std::unique_ptr<Coord> coord1 = std::make_unique<IntCoord>(13, 13);
  coord0->inc();
  *coord1 = *coord0;
  __ESBMC_assert(*coord0 == *coord1, "== after inc+assign");

  std::unique_ptr<Coord> coord2 = coord0->clone();
  __ESBMC_assert(*coord0 == *coord2, "== after clone");
  return 0;
}
