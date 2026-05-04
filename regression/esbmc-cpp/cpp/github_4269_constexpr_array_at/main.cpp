#include <array>
#include <cstdint>

constexpr std::array<uint8_t, 4> LUT = {10, 20, 30, 40};

constexpr uint8_t get_at(uint8_t i)
{
  return LUT.at(i);
}

int main()
{
  static_assert(get_at(1) == 20, "");
  static_assert(get_at(2) == 30, "");
  return 0;
}
