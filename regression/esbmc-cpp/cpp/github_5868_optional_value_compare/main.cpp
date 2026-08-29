#include <optional>
#include <cassert>

int main()
{
  std::optional<int> a(3);
  std::optional<int> big(9);
  std::optional<int> e;

  // [optional.comp.with.t]: relational comparison against a value.
  assert(a < 4);
  assert(!(a < 3));
  assert(a > 2);
  assert(a <= 3);
  assert(a >= 3);
  assert(2 < a);
  assert(4 > a);

  // A disengaged optional orders below every value.
  assert(e < 0);
  assert(!(e > 0));
  assert(0 > e);

  // [optional.relops]
  assert(a < big);
  assert(big > a);
  assert(e < a);
  assert(!(a < e));
  assert(a <= a);
  assert(a >= a);

  // [optional.nullops]
  assert(!(a < std::nullopt));
  assert(std::nullopt < a);
  assert(e <= std::nullopt);
  assert(std::nullopt >= e);

  // Equality must stay unambiguous.
  assert(a == 3);
  assert(a != big);
  assert(e == std::nullopt);
  return 0;
}
