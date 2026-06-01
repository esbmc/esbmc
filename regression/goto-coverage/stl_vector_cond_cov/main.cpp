#include <vector>
#include <cassert>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  int sum = 0;
  if (v.size() > 0)
    sum = v[0];
  if (v.size() > 100)
    sum = -1;

  assert(sum == 1);
  return 0;
}
