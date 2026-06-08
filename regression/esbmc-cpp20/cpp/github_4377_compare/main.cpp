#include <cassert>
#include <compare>

std::strong_ordering compare(int a, int b)
{
  if (a < b)
    return std::strong_ordering::less;
  if (a > b)
    return std::strong_ordering::greater;
  return std::strong_ordering::equal;
}

int main()
{
  // Direct use of strong_ordering values and comparison against 0.
  auto less = compare(1, 2);
  assert(less < 0);
  assert(less != 0);
  assert(!(less > 0));

  auto eq = compare(5, 5);
  assert(eq == 0);

  auto greater = compare(7, 3);
  assert(greater > 0);

  return 0;
}
