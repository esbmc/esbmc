// esbmc/esbmc#5868 negative control: <tuple> modelled tuple_size but not the C++17 tuple_size_v
// variable template, which ESBMC's own irep2.h uses.
#include <cassert>
#include <tuple>

int main()
{
  static_assert(std::tuple_size_v<std::tuple<>> == 0, "");
  static_assert(std::tuple_size_v<std::tuple<int>> == 1, "");
  static_assert(std::tuple_size_v<std::tuple<int, char, double>> == 3, "");
  static_assert(
    std::tuple_size_v<std::tuple<int, char>> == std::tuple_size<std::tuple<int, char>>::value,
    "");

  assert((std::tuple_size_v<std::tuple<int, char, double>> == 4));
  return 0;
}
