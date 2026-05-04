// Negative reproducer for https://github.com/esbmc/esbmc/issues/4251
// std::clamp must clamp to [lo, hi]; this assertion contradicts that and
// should fail.
#include <algorithm>
#include <cstdint>

int32_t clamp_example(int32_t v)
{
  return std::clamp(v, int32_t{0}, int32_t{100});
}

int main()
{
  __ESBMC_assert(clamp_example(150) == 150, "clamp must not modify value");
  return 0;
}
