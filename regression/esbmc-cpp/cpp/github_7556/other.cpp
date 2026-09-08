#include <string>
#include <vector>

int other_sum(const std::vector<int> &v)
{
  int s = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    s += v[i];
  return s;
}
