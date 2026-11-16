#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m{{3, 30}, {1, 10}, {1, 20}};
  assert(m.size() == 2);
  assert(m.begin()->first == 1 && m.begin()->second == 10);
  assert(m[3] == 30);

  std::map<int, int> n{{1, 1}, {2, 2}, {5, 5}};
  assert(n.size() == 3 && n.begin()->first == 1 && n[5] == 5);
  return 0;
}
