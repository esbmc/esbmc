// Counterpart of github_4243_mem_init: the mem-initializer zeroes the
// elements, but an explicit write in the ctor body must win over that
// zero-init.
#include <array>
#include <cstdint>

struct B
{
  std::array<std::uint32_t, 8> bitmap_;
  B() : bitmap_()
  {
    bitmap_[0] = 7;
  }
};

int main()
{
  B x;
  __ESBMC_assert(x.bitmap_[0] == 0, "explicit write must not be zeroed");
  return 0;
}
