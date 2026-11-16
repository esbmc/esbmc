#include <algorithm>
#include <cassert>
#include <functional>

int main()
{
  int a[] = {1, 2, 3};
  int b[] = {0};
  assert(std::includes(a, a + 3, b, b));
  assert(std::includes(a, a, b, b));
  assert(std::includes(a, a + 3, b, b, std::less<int>()));

  int c[] = {1, 3};
  int d[] = {1, 1};
  int e[] = {2, 4};
  assert(std::includes(a, a + 3, c, c + 2));
  assert(!std::includes(a, a + 3, d, d + 2));
  assert(!std::includes(a, a + 3, e, e + 2));
  assert(!std::includes(a, a, c, c + 1));

  int f[] = {3, 2, 1};
  int g[] = {3, 1};
  assert(std::includes(f, f + 3, g, g + 2, std::greater<int>()));
  return 0;
}
