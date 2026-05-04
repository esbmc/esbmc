// github.com/esbmc/esbmc/issues/4245 — std::optional shim.
#include <optional>
#include <cassert>

int main()
{
  std::optional<int> empty;
  assert(!empty.has_value());
  assert(!empty);
  assert(empty == std::nullopt);

  std::optional<int> v(42);
  assert(v.has_value());
  assert(static_cast<bool>(v));
  assert(*v == 42);
  assert(v.value() == 42);
  assert(v != std::nullopt);

  std::optional<int> w = std::nullopt;
  assert(!w);
  assert(w.value_or(7) == 7);

  w = 5;
  assert(w.has_value() && *w == 5);

  std::optional<int> copy = v;
  assert(copy.has_value() && *copy == 42);

  v.reset();
  assert(!v.has_value());

  std::optional<int> engaged(9);
  engaged = std::nullopt;
  assert(!engaged.has_value());

  return 0;
}
