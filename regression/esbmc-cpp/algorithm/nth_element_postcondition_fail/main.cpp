#include <cassert>
#include <vector>
#include <algorithm>

int main()
{
  std::vector<int> v{3, 1, 2};
  std::nth_element(v.begin(), v.begin() + 1, v.end());

  // The middle element of {1,2,3} is 2, not 3.
  assert(v[1] == 3);

  return 0;
}
