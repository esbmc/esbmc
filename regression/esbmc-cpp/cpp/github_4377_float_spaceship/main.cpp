// Floating-point <=> yields std::partial_ordering, whose fourth result is
// `unordered` when either operand is NaN ([expr.spaceship]/4).
#include <compare>
#include <cassert>

int main()
{
  double a = 1.0, b = 2.0, c = 1.0;
  double nan = 0.0 / 0.0;

  auto lt = a <=> b;
  assert(lt < 0 && !(lt > 0) && !(lt == 0));

  auto eq = a <=> c;
  assert(eq == 0 && eq <= 0 && eq >= 0);

  auto gt = b <=> a;
  assert(gt > 0 && !(gt < 0));

  // every relational comparison against 0 is false for unordered
  auto un = nan <=> a;
  assert(!(un < 0));
  assert(!(un > 0));
  assert(!(un == 0));
  assert(!(un <= 0));
  assert(!(un >= 0));
  assert(un == std::partial_ordering::unordered);
  assert(!(un == std::partial_ordering::less));

  // category conversions
  std::partial_ordering ps = std::strong_ordering::less;
  assert(ps < 0);
  std::partial_ordering pw = std::weak_ordering::greater;
  assert(pw > 0);
  return 0;
}
