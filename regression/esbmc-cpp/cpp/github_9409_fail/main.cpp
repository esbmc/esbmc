#include <cassert>
#include <map>

int main()
{
  std::map<int, int> m{{3, 30}, {1, 10}, {1, 20}};
  assert(m.size() == 3 && m.begin()->first == 3);
  return 0;
}
