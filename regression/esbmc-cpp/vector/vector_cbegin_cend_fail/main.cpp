#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  int s = 0;
  for (std::vector<int>::const_iterator it = v.cbegin(); it != v.cend(); ++it)
    s += *it;
  assert(s == 99);
  return 0;
}
