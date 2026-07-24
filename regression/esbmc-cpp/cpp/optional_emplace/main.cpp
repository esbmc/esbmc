#include <optional>
#include <cassert>

int main()
{
  std::optional<int> o;
  int &r = o.emplace(7);
  assert(o.has_value() && *o == 7);
  r = 8; // returned reference aliases the contained value
  assert(*o == 8);

  o.emplace(3); // emplace over an engaged optional
  assert(*o == 3);

  std::optional<int> a = 1, b = 2;
  a.swap(b);
  assert(*a == 2 && *b == 1);

  std::optional<int> c = 5, d;
  c.swap(d); // swap engaged with empty
  assert(!c.has_value() && *d == 5);

  auto m = std::make_optional(42);
  assert(m.has_value() && *m == 42);

  return 0;
}
