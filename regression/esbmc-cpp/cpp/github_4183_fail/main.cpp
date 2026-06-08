#include <array>
#include <cassert>
#include <cstdint>

int main()
{
  std::array<uint8_t, 4> a{};
  a[0] = 7;
  assert(a[0] == 0);
  return a[0];
}
