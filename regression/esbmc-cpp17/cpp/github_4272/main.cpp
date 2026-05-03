// github.com/esbmc/esbmc/issues/4272 — std::tuple must be a literal type
// in C++17 so that constexpr std::array<std::tuple<...>, N> compiles and
// retains its element values.

#include <array>
#include <tuple>
#include <cstdint>
#include <cassert>

constexpr std::array<std::tuple<uint8_t, uint8_t>, 2> TABLE = {
  {std::tuple<uint8_t, uint8_t>{1, 2}, std::tuple<uint8_t, uint8_t>{3, 4}}};

int main()
{
  assert(std::get<0>(TABLE[0]) == 1);
  assert(std::get<1>(TABLE[0]) == 2);
  assert(std::get<0>(TABLE[1]) == 3);
  assert(std::get<1>(TABLE[1]) == 4);
  return 0;
}
