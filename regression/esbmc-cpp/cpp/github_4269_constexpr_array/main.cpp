#include <array>
#include <cstdint>

constexpr std::array<uint8_t, 4> LUT = {10, 20, 30, 40};

constexpr uint8_t get(uint8_t i)
{
  return LUT[i];
}

int main()
{
  static_assert(get(0) == 10, "");
  static_assert(get(3) == 40, "");
  return 0;
}
