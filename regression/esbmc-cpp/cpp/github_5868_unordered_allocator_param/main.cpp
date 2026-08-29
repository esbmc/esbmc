#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>
#include <cassert>

int main()
{
  // [unord.map.overview] / [unord.set.overview]: five and four parameters.
  std::unordered_map<
    int, int, std::hash<int>, std::equal_to<int>,
    std::allocator<std::pair<const int, int>>> m;
  m[1] = 10;
  assert(m[1] == 10);
  assert(m.size() == 1);

  std::unordered_set<
    int, std::hash<int>, std::equal_to<int>, std::allocator<int>> s;
  s.insert(4);
  assert(s.count(4) == 1);

  // The defaults must still work.
  std::unordered_map<int, int> dm;
  dm[3] = 30;
  assert(dm[3] == 30);
  std::unordered_set<int> ds;
  ds.insert(6);
  assert(ds.count(6) == 1);
  return 0;
}
