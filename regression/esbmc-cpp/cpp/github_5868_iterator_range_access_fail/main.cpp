#include <iterator>
#include <cassert>

int main()
{
  int a[4] = {1, 2, 3, 4};
  // [iterator.range]: size() on an array is its extent, not its first element.
  assert(std::size(a) == 1);
  return 0;
}
