// github.com/esbmc/esbmc/issues/4272 — negative test: a constexpr array of
// tuples must hold the values it was initialized with, so a wrong assertion
// must fail (otherwise the literal-type fix would mask incorrect storage).

#include <array>
#include <tuple>
#include <cstdint>
#include <cassert>

constexpr std::array<std::tuple<uint8_t, uint8_t>, 2> TABLE = {
  {std::tuple<uint8_t, uint8_t>{1, 2}, std::tuple<uint8_t, uint8_t>{3, 4}}};

int main()
{
  assert(std::get<1>(TABLE[0]) == 99);
  return 0;
}
