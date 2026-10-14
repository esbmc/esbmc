// Negative companion to github_6330_data: data()[0] is 1, so asserting a wrong
// value is violated.
#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v = {1, 2, 3};
  assert(v.data()[0] == 42); // wrong on purpose: it is 1
  return 0;
}
