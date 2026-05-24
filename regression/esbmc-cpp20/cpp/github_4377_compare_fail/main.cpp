#include <cassert>
#include <compare>

int main()
{
  std::strong_ordering o = std::strong_ordering::less;
  // o < 0 holds; the assertion below must fail.
  assert(o > 0);
  return 0;
}
