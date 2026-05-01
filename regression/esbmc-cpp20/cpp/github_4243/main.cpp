#include <array>
#include <cstdint>

int main()
{
  std::array<int32_t, 4> buf{};
  __ESBMC_assert(
    buf[0] == 0, "value-init must zero-initialise std::array elements");
  return 0;
}
