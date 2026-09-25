#include <set>
#include <cassert>

int main()
{
  std::set<int> s;
  s.insert(1);
  s.insert(2);
  int sum = 0;
  for (std::set<int>::const_iterator it = s.cbegin(); it != s.cend(); ++it)
    sum += *it;
  assert(sum == 99); // wrong
  return 0;
}
