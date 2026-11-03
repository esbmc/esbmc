#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.reserve(10);
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.insert(v.begin(), v[1]);
  assert(v[0] != 2);
  return 0;
}
