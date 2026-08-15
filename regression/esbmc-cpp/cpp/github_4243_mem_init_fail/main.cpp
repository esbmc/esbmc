#include <array>
#include <cstdint>

struct B
{
  std::array<std::uint32_t, 8> bitmap_;
  B() : bitmap_() {}
};

int main()
{
  B x;
  __ESBMC_assert(
    x.bitmap_[0] != 0,
    "value-init must zero-initialise std::array members");
  return 0;
}
