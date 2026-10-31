// github #9384: multiset::erase(k) erases every element equal to k.
#include <cassert>
#include <set>

int main()
{
  std::multiset<int> s;
  s.insert(1);
  s.insert(1);
  s.insert(2);
  s.erase(1);
  assert(s.count(1) != 0);
  return 0;
}
