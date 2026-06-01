// Reproducer for https://github.com/esbmc/esbmc/issues/4251
// std::clamp must be available from the bundled <algorithm> header.
#include <algorithm>
#include <cstdint>

int32_t clamp_example(int32_t v)
{
  return std::clamp(v, int32_t{0}, int32_t{100});
}

int main()
{
  __ESBMC_assert(clamp_example(150) == 100, "clamp hi");
  __ESBMC_assert(clamp_example(-10) == 0, "clamp lo");
  __ESBMC_assert(clamp_example(50) == 50, "clamp mid");
  return 0;
}
