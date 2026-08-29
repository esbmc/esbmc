// Negative counterpart of github_3387_containers_c++03: the vector holds 2
// elements, so the program is really verified rather than passing vacuously.
#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  assert(v.size() == 3);
  return 0;
}
