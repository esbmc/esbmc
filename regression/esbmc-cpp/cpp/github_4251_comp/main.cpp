// Comparator overload for https://github.com/esbmc/esbmc/issues/4251
// Exercises std::clamp(v, lo, hi, comp) — i.e. the 4-argument template.
#include <algorithm>
#include <cstdint>
#include <functional>

int32_t clamp_reverse(int32_t v)
{
  // With std::greater<>, the ordering is inverted: lo and hi roles flip.
  return std::clamp(v, int32_t{100}, int32_t{0}, std::greater<int32_t>{});
}

int main()
{
  __ESBMC_assert(clamp_reverse(150) == 100, "clamp_reverse hi");
  __ESBMC_assert(clamp_reverse(-10) == 0, "clamp_reverse lo");
  __ESBMC_assert(clamp_reverse(50) == 50, "clamp_reverse mid");
  return 0;
}
