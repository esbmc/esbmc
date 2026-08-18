// [meta.unary.prop]: the nothrow copy/destroy traits. boost's iterator_facade
// names is_nothrow_copy_constructible, which the model did not have -- only the
// move forms. See #5868.
#include <type_traits>
#include <cassert>

struct thrower
{
  thrower(const thrower &);
  thrower &operator=(const thrower &);
  ~thrower();
};

struct plain
{
  int a;
};

int main()
{
  static_assert(std::is_nothrow_copy_constructible<int>::value, "int");
  static_assert(std::is_nothrow_copy_constructible<plain>::value, "plain");
  static_assert(!std::is_nothrow_copy_constructible<thrower>::value, "thrower");
  static_assert(std::is_nothrow_copy_assignable<plain>::value, "assign");
  static_assert(!std::is_nothrow_copy_assignable<thrower>::value, "assign2");
  static_assert(std::is_nothrow_destructible<plain>::value, "dtor");
  static_assert(std::is_nothrow_copy_constructible_v<int>, "v");
  assert(1);
  return 0;
}
