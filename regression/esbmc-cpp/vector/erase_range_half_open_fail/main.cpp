#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v{1, 2, 3, 4};
  v.erase(v.begin() + 1, v.begin() + 3);

  // The range is half-open, so 4 survives and 2 does not.
  assert(v[1] == 3);

  return 0;
}
