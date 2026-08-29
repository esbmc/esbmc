#include <cassert>
#include <iterator>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  // rbegin starts at the last element, which is 3.
  assert(*std::rbegin(v) == 1);

  return 0;
}
