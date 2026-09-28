#include <set>
#include <cassert>

int main()
{
  std::set<int> a, b;
  a.insert(1);
  a.insert(3);
  b.insert(2);

  // Ordering by size instead of lexicographically would make {2} < {1,3}.
  assert(b < a);
  return 0;
}
