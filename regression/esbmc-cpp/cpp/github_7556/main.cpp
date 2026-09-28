// Merging a second C++ translation unit went through
// DeclContext::localUncachedLookup once per imported decl, which is quadratic
// in the destination context and did not finish on real multi-file projects.
// This pins that a two-TU C++ merge still produces a working program (#7556).
#include <cassert>
#include <vector>

int other_sum(const std::vector<int> &v);

int main()
{
  std::vector<int> v;
  v.push_back(2);
  v.push_back(3);
  assert(other_sum(v) == 5);
  return 0;
}
