#include <cassert>
#include <iterator>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  // [iterator.range]: rbegin walks the container backwards.
  assert(*std::rbegin(v) == 3);

  int n = 0, sum = 0;
  for (auto i = std::rbegin(v); i != std::rend(v); ++i)
  {
    sum += *i;
    n++;
  }
  assert(n == 3);
  assert(sum == 6);

  assert(*std::crbegin(v) == 3);

  const std::vector<int> &cv = v;
  assert(*std::rbegin(cv) == 3);

  return 0;
}
