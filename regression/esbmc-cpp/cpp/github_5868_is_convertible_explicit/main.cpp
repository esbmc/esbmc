#include <type_traits>

struct target
{
};
struct explicit_src
{
  explicit operator target() const;
};
struct implicit_src
{
  operator target() const;
};

int main()
{
  static_assert(
    !std::is_convertible<explicit_src, target>::value,
    "an explicit operator is not an implicit conversion");
  static_assert(
    !std::is_convertible<const explicit_src &, target>::value,
    "same through a const reference");
  static_assert(
    std::is_convertible<implicit_src, target>::value,
    "a non-explicit operator still converts");

  static_assert(std::is_convertible<int, long>::value, "arithmetic");
  static_assert(!std::is_convertible<int, target>::value, "unrelated");
  static_assert(std::is_convertible<void, void>::value, "[meta.rel]: void");
  static_assert(!std::is_convertible<void, int>::value, "[meta.rel]: void");
  static_assert(!std::is_convertible<int, int[3]>::value, "[meta.rel]: array");
  return 0;
}
