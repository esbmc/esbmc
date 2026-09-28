// Negative counterpart of github_3387_compare_pre_cxx20: the translation unit
// is really verified after the <compare> include, not silently skipped.
#include <compare>
#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(3);
  v.push_back(1);
  assert(v[0] < v[1]);
  return 0;
}
