#include <type_traits>

struct trivial
{
  int a;
};

struct throwing
{
  throwing()
  {
    throw 1;
  }
};

struct nontrivial
{
  nontrivial()
  {
  }
  ~nontrivial()
  {
  }
};

int main()
{
  // [meta.unary.prop]
  static_assert(std::is_nothrow_default_constructible<int>::value, "int");
  static_assert(std::is_nothrow_default_constructible<trivial>::value, "pod");
  static_assert(
    !std::is_nothrow_default_constructible<throwing>::value, "throws");

  static_assert(std::is_nothrow_constructible<int>::value, "0 args");
  static_assert(std::is_nothrow_constructible<int, int>::value, "1 arg");
  static_assert(std::is_nothrow_assignable<int &, int>::value, "assign");

  static_assert(std::is_trivially_constructible<trivial>::value, "trivial");
  static_assert(!std::is_trivially_constructible<nontrivial>::value, "nontrivial");
  static_assert(std::is_trivially_copy_constructible<trivial>::value, "copy");
  static_assert(std::is_trivially_move_constructible<trivial>::value, "move");
  static_assert(std::is_trivially_assignable<int &, int>::value, "assignable");
  static_assert(std::is_trivially_copy_assignable<trivial>::value, "copy assign");
  static_assert(std::is_trivially_move_assignable<trivial>::value, "move assign");

  static_assert(std::is_nothrow_default_constructible_v<int>, "_v alias");
  static_assert(std::is_trivially_constructible_v<trivial>, "_v alias");
  return 0;
}
