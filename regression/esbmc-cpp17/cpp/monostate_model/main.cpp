// [variant.monostate]: a unit type, so a variant whose first alternative is
// not default-constructible can still be default-constructed. The variant
// model listed it as out of scope (issue #5868).
#include <variant>

int main()
{
  std::monostate a, b;
  __ESBMC_assert(a == b, "all monostate values compare equal");
  __ESBMC_assert(!(a != b), "and none unequal");
  __ESBMC_assert(!(a < b) && !(a > b), "neither orders before the other");
  __ESBMC_assert(a <= b && a >= b, "and both are <= and >=");

  std::variant<std::monostate, int> v;
  __ESBMC_assert(v.index() == 0, "a variant defaults to monostate");
  __ESBMC_assert(
    std::holds_alternative<std::monostate>(v), "and holds it");

  v = 3;
  __ESBMC_assert(v.index() == 1, "assignment moves the discriminator");
  __ESBMC_assert(std::get<int>(v) == 3, "and stores the value");
  __ESBMC_assert(
    !std::holds_alternative<std::monostate>(v), "no longer the unit type");

  // Note: monostate's other classic use -- letting a variant whose first
  // alternative is not default-constructible still be default-constructed --
  // is not reachable here. The model gives each alternative its own member
  // and value-initialises all of them, so every alternative must be
  // default-constructible regardless. That is a property of the flat-member
  // design the header documents, not of monostate.
  return 0;
}
