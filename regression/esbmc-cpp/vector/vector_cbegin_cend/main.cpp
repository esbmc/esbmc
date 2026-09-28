#include <cassert>
#include <vector>

int main()
{
  std::vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  int fwd = 0;
  for (std::vector<int>::const_iterator it = v.cbegin(); it != v.cend(); ++it)
    fwd += *it;
  assert(fwd == 6);

  int rev = 0;
  for (std::vector<int>::const_reverse_iterator it = v.crbegin();
       it != v.crend();
       ++it)
    rev = rev * 10 + *it;
  assert(rev == 321);

  return 0;
}
