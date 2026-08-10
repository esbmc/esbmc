#include <cassert>
#include <vector>

int main()
{
  std::vector<int> a{2}, b{1, 3};

  // Ordering is lexicographic, not by size: 2 > 1 makes a the greater range
  // even though it is the shorter one.
  assert(a < b);

  return 0;
}
