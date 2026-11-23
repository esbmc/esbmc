#include <flat_set>
#include <cassert>
#include <vector>

int main()
{
  std::flat_set<int> s;
  s.insert(3);
  s.insert(1);
  std::vector<int> v = std::move(s).extract();
  s.replace(std::move(v));
  assert(s.contains(2));
  return 0;
}
