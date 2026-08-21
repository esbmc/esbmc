#include <type_traits>

struct plain
{
};
struct no_default
{
  no_default(int)
  {
  }
};
struct no_move
{
  no_move(const no_move &)
  {
  }
  no_move &operator=(const no_move &)
  {
    return *this;
  }
};

int main()
{
  static_assert(std::is_default_constructible<plain>::value, "plain");
  static_assert(std::is_default_constructible<int>::value, "int");
  static_assert(!std::is_default_constructible<no_default>::value, "no_default");
  static_assert(std::is_default_constructible_v<plain>, "_v alias");

  static_assert(std::is_move_constructible<plain>::value, "move ctor");
  static_assert(std::is_move_constructible<no_move>::value, "copy ctor serves");
  static_assert(std::is_move_constructible_v<int>, "_v alias");

  static_assert(std::is_move_assignable<plain>::value, "move assign");
  static_assert(std::is_move_assignable_v<int>, "_v alias");
  static_assert(!std::is_move_assignable<const int>::value, "const");
  return 0;
}
