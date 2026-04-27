#include <array>
#include <cassert>
#include <cstdint>

int main()
{
  std::array<uint8_t, 4> a{};
  a[0] = 42;
  a[3] = 7;
  assert(a[0] == 42);
  assert(a[3] == 7);
  return 0;
}
