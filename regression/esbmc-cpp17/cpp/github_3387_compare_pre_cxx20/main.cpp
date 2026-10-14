// Including <compare> in C++17 must be a no-op rather than a parse error, and
// must not disturb the rest of the translation unit (github #3387).
#include <compare>
#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(3);
  v.push_back(1);
  assert(v.size() == 2);
  assert(v[0] > v[1]);
  return 0;
}
