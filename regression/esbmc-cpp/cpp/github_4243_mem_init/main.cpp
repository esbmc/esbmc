#include <array>
#include <cstdint>

struct B
{
  std::array<std::uint32_t, 8> bitmap_;
  B() : bitmap_() {} // parenthesized value-init in a ctor mem-initializer
};

int main()
{
  B x;
  for (std::size_t i = 0; i < x.bitmap_.size(); ++i)
    __ESBMC_assert(
      x.bitmap_[i] == 0,
      "value-init must zero-initialise std::array members");
  return 0;
}
