// Anti-vacuity twin of type_traits_nothrow_copy: the trait distinguishes a
// throwing copy from a trivial one, so claiming otherwise must fail.
#include <type_traits>
#include <cassert>

struct thrower
{
  thrower(const thrower &);
};

int main()
{
  assert(std::is_nothrow_copy_constructible<thrower>::value);
  return 0;
}
