// Test: operator== returns false for two IntCoords with different values.
// Verifies that the virtual equals() dispatch correctly distinguishes
// (17,17) from (13,13) — assertion must fail.
#include <memory>

class Coord
{
public:
  virtual ~Coord() = default;
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
  // no assign — coords remain (17,17) vs (13,13)
  __ESBMC_assert(*coord0 == *coord1, "should fail: 17,17 != 13,13");
  return 0;
}
