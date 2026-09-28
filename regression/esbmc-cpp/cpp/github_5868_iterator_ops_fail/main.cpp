// Negative counterpart of github_5868_iterator_ops: distance() now computes a
// real length instead of returning an unconstrained value, so a wrong claim is
// refuted rather than being satisfiable.
#include <iterator>
#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  assert(std::distance(v.begin(), v.end()) == 2);
  return 0;
}
