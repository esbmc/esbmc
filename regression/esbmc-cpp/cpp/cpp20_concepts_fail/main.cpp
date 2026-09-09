#include <concepts>
#include <cassert>

template <std::integral T> T twice(T v) { return v + v; }
template <std::floating_point T> T half(T v) { return v / 2; }

struct Plain { int x = 0; bool operator==(const Plain &) const = default; };

int main() {
  static_assert(std::same_as<int, int>);
  static_assert(!std::same_as<int, long>);
  static_assert(std::integral<int> && !std::integral<double>);
  static_assert(std::signed_integral<int> && std::unsigned_integral<unsigned>);
  static_assert(std::floating_point<double>);
  static_assert(std::convertible_to<int, long>);
  static_assert(std::destructible<int>);
  static_assert(std::default_initializable<Plain>);
  static_assert(std::equality_comparable<int>);
  static_assert(std::totally_ordered<int>);
  static_assert(std::movable<Plain>);
  static_assert(std::copyable<Plain>);
  static_assert(std::semiregular<Plain>);
  static_assert(std::regular<int>);
  assert(twice(3) == 7);
  assert(half(4.0) == 2.0);
  return 0;
}
