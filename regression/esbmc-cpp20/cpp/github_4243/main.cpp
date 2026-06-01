#include <array>
#include <cstdint>

int main()
{
  std::array<int32_t, 4> buf{};
  for (std::size_t i = 0; i < buf.size(); ++i)
    __ESBMC_assert(
      buf[i] == 0, "value-init must zero-initialise std::array elements");
  return 0;
}
