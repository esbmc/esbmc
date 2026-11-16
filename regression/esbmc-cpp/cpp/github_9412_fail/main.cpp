#include <cassert>
#include <set>

int main()
{
  int a[3] = {1, 1, 2};
  std::set<int> s(a, a + 3);
  assert(s.size() != 2);
  return 0;
}
