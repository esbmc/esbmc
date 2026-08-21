#include <array>
#include <cstdint>

struct B
{
  std::array<std::uint32_t, 8> bitmap_;
  B() : bitmap_()
  {
  } // parenthesized value-init in a ctor mem-initializer
};

struct M
{
  std::uint32_t v;
  M() : v(7)
  {
  } // user-provided default ctor: must keep the call path
};

struct W
{
  M m; // implicit, non-trivial default ctor: must keep the call path
};

struct U
{
  M m;
  W w;
  U() : m(), w()
  {
  } // parenthesized mem-initializers must still call
};

int main()
{
  B x;
  for (std::size_t i = 0; i < x.bitmap_.size(); ++i)
    __ESBMC_assert(
      x.bitmap_[i] == 0, "value-init must zero-initialise std::array members");

  U u;
  __ESBMC_assert(u.m.v == 7, "user-provided ctor still runs");
  __ESBMC_assert(u.w.m.v == 7, "non-trivial implicit ctor still runs");
  return 0;
}
