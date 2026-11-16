#include <algorithm>
#include <cassert>

int main()
{
  int a[] = {1, 2, 3};
  int b[] = {0};
  assert(!std::includes(a, a + 3, b, b));
  return 0;
}
