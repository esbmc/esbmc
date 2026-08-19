#include <optional>
#include <cassert>

int main()
{
  std::optional<int> a(3);
  std::optional<int> b(3);
  std::optional<int> e;

  // [optional.comp.with.t]: comparison against a value.
  assert(a == 3);
  assert(3 == a);
  assert(a != 4);
  assert(4 != a);
  assert(!(e == 3));
  assert(e != 3);

  // [optional.relops] and the nullopt overloads must stay unambiguous.
  assert(a == b);
  assert(!(a != b));
  assert(a != e);
  assert(e == std::nullopt);
  assert(a != std::nullopt);
  return 0;
}
